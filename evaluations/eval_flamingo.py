import argparse
import torch
import datasets
import datasets.utils
import os
import numpy as np
import time
import random
import XTransferBench
import XTransferBench.zoo
from tqdm import tqdm
import json
import torch.nn.functional as F
from collections import defaultdict
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from open_flamingo import create_model_and_transforms
from torchvision import transforms 
from functools import partial
from ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from nltk.translate.bleu_score import sentence_bleu

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='XTransferBench')

parser.add_argument('--seed', type=int, default=7, help='seed')
# Dataset 
parser.add_argument('--task', default='captioning', type=str, help='Task for evaluation')
parser.add_argument('--dataset_name', default='coco', type=str, help='Dataset for evaluation')
parser.add_argument('--train_dataset', default='CaptionDataset', type=str, help='Dataset for evaluation')
parser.add_argument('--train_data_path', default='/path/to/coco/train2017', type=str, help='Path to the dataset')
parser.add_argument('--train_questions_path', default='/path/to/OK-VQA/OpenEnded_mscoco_train2014_questions.json', type=str, help='Path to the VQA questions')
parser.add_argument('--train_annotations_path', default='/path/to/coco/annotations/captions_train2017.json', type=str, help='Path to the anotations')
parser.add_argument('--train_image_dir_path', default='/path/to/coco/train2017', type=str, help='Path to the imgage directory')
parser.add_argument('--test_dataset', default='CaptionDataset', type=str, help='Dataset for evaluation')
parser.add_argument('--test_data_path', default='/path/to/coco/val2017', type=str, help='Path to the dataset')
parser.add_argument('--test_questions_path', default='/path/to/OK-VQA/OpenEnded_mscoco_val2014_questions.json', type=str, help='Path to the VQA questions')
parser.add_argument('--test_annotations_path', default='/path/to/coco/annotations/captions_val2017.json', type=str, help='Path to the anotations')
parser.add_argument('--test_image_dir_path', default='/path/to/coco/val2017', type=str, help='Path to the imgage directory')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for the evaluation')
# Attacker
parser.add_argument('--threat_model', default='linf_non_targeted', type=str, help='The type of the attacker, see XTransferBench documentation for more details')
parser.add_argument('--attacker_name', default='xtransfer_large_linf_eps12_non_targeted', type=str, help='The name of the attacker, see XTransferBench documentation for more details')
parser.add_argument("--epsilon", type=int, default=16, help="Epsilon for the L_inf attack")

def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    """
    Sample random demonstrations from the query set.
    """
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]

def get_query_set(train_dataset, query_set_size):
    """
    Get a subset of the training dataset to use as the query set.
    """
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]

def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

def compute_bleu(result_path, target_text, key, weights=(0.25, 0.25, 0.25, 0.25)):
    correct = 0
    with open(result_path) as f:
        results = json.load(f)
        for item in results:
            score = sentence_bleu([target_text], item[key], weights=weights)
            correct += score
    return correct / len(results)

def compute_cider(
    result_path,
    annotations_path,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    return coco_eval.eval

def postprocess_captioning_generation(predictions):
    return predictions.split("Output", 1)[0]

def get_caption_prompt(caption=None):
    return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

def get_vqa_prompt(question, answer=None):
    return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

def _prepare_images(batch, image_processor):
    
    images_per_example = max(len(x) for x in batch)
    batch_images = None
    for iexample, example in enumerate(batch):
        for iimage, image in enumerate(example):
            preprocessed = image_processor(image)
            if batch_images is None:
                batch_images = torch.zeros(
                    (len(batch), images_per_example, 1) + preprocessed.shape,
                    dtype=preprocessed.dtype,
                )
            batch_images[iexample, iimage, 0] = preprocessed
    if batch_images is not None:
        batch_images = batch_images.to(
            device, non_blocking=True
        )
    return batch_images

def _prepare_text(batch, tokenizer, padding="longest", truncation=True, max_length=2000):
    encodings = tokenizer(
        batch,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
        max_length=max_length,
    )
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(
        device, non_blocking=True
    )
    return input_ids, attention_mask.bool()

def add_pattern(img, attacker):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return attacker.attack(img)[0]

def _convert_to_rgb(image):
    return image.convert('RGB')

def eval_captioning(train_dataset, test_dataset, model, image_processor, tokenizer,
                    num_shots=0, min_generation_length=0, max_generation_length=20, 
                    num_beams=3, length_penalty=0, scaler=None, tag='clean'):    
    batch_size = args.batch_size
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
    )

    query_set = get_query_set(train_dataset, 2048)
    effective_num_shots = 2 if num_shots <= 0 else num_shots

    predictions = defaultdict()
    for batch in tqdm(test_dataloader):

        batch_demo_samples = sample_batch_demos_from_query_set(
                    query_set, effective_num_shots, len(batch["image"])
                )
        
        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    get_caption_prompt(caption=x["caption"].strip()) + "\n"
                    for x in batch_demo_samples[i]
                ]
            )
            
            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")
            
            batch_text.append(context_text + get_caption_prompt())

        batch_images = _prepare_images(batch_images, image_processor)
        input_ids, attention_mask = _prepare_text(batch_text, tokenizer)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.inference_mode():
                outputs = model.generate(
                    batch_images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
         # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]
        
        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": new_predictions[i],
            }

    all_predictions = predictions

    # save the predictions to a temporary file
    results_path = "{}_{}_{}_shots.json".format(args.dataset_name, num_shots, tag)
    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": all_predictions[k]["caption"]}
                    for k in all_predictions
                ],
                indent=4,
            )
        )
    return results_path

def eval_vqa(train_dataset, test_dataset, model, image_processor, tokenizer,
             num_shots=0, min_generation_length=0, max_generation_length=20, 
             num_beams=3, length_penalty=0, scaler=None, tag='clean'):    
    batch_size = args.batch_size 
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )

    query_set = get_query_set(train_dataset, 2048)
    effective_num_shots = 2 if num_shots <= 0 else num_shots

    predictions = []
    for batch in tqdm(test_dataloader):

        batch_demo_samples = sample_batch_demos_from_query_set(
                    query_set, effective_num_shots, len(batch["image"])
                )
        
        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
                
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    + "\n"
                    for x in batch_demo_samples[i]
                ]
            )
            
            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + get_vqa_prompt(question=batch["question"][i])
            )

        batch_images = _prepare_images(batch_images, image_processor)
        input_ids, attention_mask = _prepare_text(batch_text, tokenizer)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.inference_mode():
                outputs = model.generate(
                    batch_images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
         # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        process_function = (
            postprocess_ok_vqa_generation
            if args.dataset_name == "ok_vqa"
            else postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})

    all_predictions = predictions
    
    # save the predictions to a temporary file
    results_path = "{}_{}_{}_shots.json".format(args.dataset_name, num_shots, tag)
    with open(results_path, "w") as f:
        f.write(json.dumps(all_predictions, indent=4))
    
    acc = compute_vqa_accuracy(
            results_path,
            args.test_questions_path,
            args.test_annotations_path,
        )

    return acc

def main():
    # Prepare Flamingo Model
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        
    #     cache_dir="checkpoints"  # Defaults to ~/.cache
    )
    tokenizer.padding_side = "left"
    checkpoint = torch.load("openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt", 
                            map_location=device)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)      
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False 
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    scaler = torch.cuda.amp.GradScaler() 

    attacker = XTransferBench.zoo.load_attacker(args.threat_model, args.attacker_name)
    attacker = attacker.to(device)
    
    if hasattr(attacker, 'delta'):
        # L_inf attack
        print('L_inf attack with epsilon: {}'.format(args.epsilon))
        attacker.interpolate_epsilon(args.epsilon/255)

    adversarial_image_processor = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
        transforms.ToTensor(),
        partial(add_pattern, attacker=attacker),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                            std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_set = datasets.utils.dataset_options[args.train_dataset](
        args.train_image_dir_path, transform=None, is_test=False, kwargs={
            "image_train_dir_path": args.train_image_dir_path,
            "image_val_dir_path": args.test_image_dir_path,
            "train_question_path": args.train_questions_path,
            "train_annotations_path": args.train_annotations_path,
            "val_question_path": args.test_questions_path,
            "val_annotations_path": args.test_annotations_path,
            "dataset_name": args.dataset_name,
            "train_dataset_name": args.dataset_name,
            "val_dataset_name": args.dataset_name,
        })
    test_set = datasets.utils.dataset_options[args.test_dataset](
        args.test_image_dir_path, transform=None, is_test=True, kwargs={
            "image_train_dir_path": args.train_image_dir_path,
            "image_val_dir_path": args.test_image_dir_path,
            "train_question_path": args.train_questions_path,
            "train_annotations_path": args.train_annotations_path,
            "val_question_path": args.test_questions_path,
            "val_annotations_path": args.test_annotations_path,
            "dataset_name": args.dataset_name,
            "train_dataset_name": args.dataset_name,
            "val_dataset_name": args.dataset_name,
        })

    if args.task == 'captioning':
        # Eval Clean Performance
        zero_shot_clean_result_path = eval_captioning(train_set, test_set, model, image_processor, tokenizer, num_shots=0, scaler=scaler, tag='clean')
        
        # Eval Adversarial Performance
        adv_zero_shot_result_path = eval_captioning(train_set, test_set, model, adversarial_image_processor, tokenizer, num_shots=0, scaler=scaler, tag='adv')
        
        # Compute 
        clean_zero_shot_cider = compute_cider(zero_shot_clean_result_path, args.test_annotations_path)['CIDEr']
        print("\033[33mClean Zero Shot Cider: {:.4f}\033[0m".format(clean_zero_shot_cider))

        untargeted_adv_zero_shot_cider = compute_cider(adv_zero_shot_result_path, args.test_annotations_path)['CIDEr']
        print("\033[33mUntargeted Adv Zero Shot Cider: {:.4f}\033[0m".format(untargeted_adv_zero_shot_cider))

        if hasattr(attacker, 'target_text') and attacker.target_text is not None:
            targeted_adv_zero_shot_bleu4 = compute_bleu(adv_zero_shot_result_path, attacker.target_text, 'caption', weights=(0.25, 0.25, 0.25, 0.25))
            print("\033[33mTargeted Adv Zero Shot Clean Bleu4: {:.4f}\033[0m".format(targeted_adv_zero_shot_bleu4))

            targeted_adv_zero_shot_bleu1 = compute_bleu(adv_zero_shot_result_path, attacker.target_text, 'caption', weights=(1,))
            print("\033[33mTargeted Adv Zero Shot Clean Bleu1: {:.4f}\033[0m".format(targeted_adv_zero_shot_bleu1))

            untargeted_asr = (clean_zero_shot_cider - untargeted_adv_zero_shot_cider) / clean_zero_shot_cider
            print("\033[33mUntargeted ASR Cider: {:.4f}\033[0m".format(untargeted_asr))

    elif args.task == 'vqa':
        # Eval Clean Performance
        zero_shot_clean_acc = eval_vqa(train_set, test_set, model, image_processor, tokenizer, num_shots=0, scaler=scaler, tag='clean')
        # Eval Adversarial Performance
        adv_zero_shot_acc = eval_vqa(train_set, test_set, model, adversarial_image_processor, tokenizer, num_shots=0, scaler=scaler, tag='adv')
        
        # Compute 
        print("\033[33mClean Zero Shot VQA Acc: {:.4f}\033[0m".format(zero_shot_clean_acc))
        print("\033[33mUntargeted Adv Zero Shot VQA Acc: {:.4f}\033[0m".format(adv_zero_shot_acc))
        print("\033[33mUntargeted ASR: {:.4f}\033[0m".format((zero_shot_clean_acc - adv_zero_shot_acc) / zero_shot_clean_acc))

    return


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seed = args.seed
    args.gpu = device

    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    print(payload)
    