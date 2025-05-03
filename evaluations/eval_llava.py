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
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

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
parser.add_argument("--conv-mode", type=str, default="llava_v1")
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
parser.add_argument('--test_annotations_path', default='/path/to/captions_val2017.json', type=str, help='Path to the anotations')
parser.add_argument('--test_image_dir_path', default='/path/to/coco/val2017', type=str, help='Path to the imgage directory')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for the evaluation')
# Attacker
parser.add_argument('--threat_model', default='linf_non_targeted', type=str, help='The type of the attacker, see XTransferBench documentation for more details')
parser.add_argument('--attacker_name', default='xtransfer_large_linf_eps12_non_targeted', type=str, help='The name of the attacker, see XTransferBench documentation for more details')
parser.add_argument("--epsilon", type=int, default=16, help="Epsilon for the L_inf attack")

def _convert_to_rgb(image):
    return image.convert('RGB')

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

def add_pattern(img, attacker):
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = attacker.attack(img)[0]
    img = transforms.ToPILImage()(img)
    return img

def eval_captioning(test_dataset, model, image_processor, tokenizer,
                    num_shots=0, max_generation_length=20, 
                    num_beams=1, scaler=None, tag='clean'):    
    batch_size = 1 # LLAVA only supports batch size 1
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn
    )

    predictions = defaultdict()
    for batch in tqdm(test_dataloader):
        # image_tensor = image_processor(batch["image"])

        image_tensor = []
        if len(batch["image"]) == 0:
            continue
        for i in range(len(batch["image"])):
            image = image_processor(_convert_to_rgb(batch["image"][i]))
            image_tensor.append(image)
        image_tensor = torch.stack(image_tensor, dim=0).to(device)
        
        qs = "Provide a short caption for this image."

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    num_beams=num_beams,
                    max_new_tokens=max_generation_length,
                    use_cache=False
                )

        # Extract only the new gnerated tokens
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": outputs[i],
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

def eval_vqa(test_dataset, model, image_processor, tokenizer,
             num_shots=0, max_generation_length=20, 
             num_beams=1, scaler=None, tag='clean'):    
    batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
    )

    predictions = []
    for batch in tqdm(test_dataloader):

        image_tensor = []
        if len(batch["image"]) == 0:
            continue
        for i in range(len(batch["image"])):
            image = image_processor(_convert_to_rgb(batch["image"][i]))
            image_tensor.append(image)
        image_tensor = torch.stack(image_tensor, dim=0).to(device)
        
        if 'ok_vqa' in args.dataset_name:
            prompt_suffix = "\nAnswer the question using a single word or phrase."
        else:
            prompt_suffix = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        qs = batch['question'][0] + prompt_suffix

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    num_beams=num_beams,
                    max_new_tokens=max_generation_length,
                    use_cache=False
                )

        # Extract only the new gnerated tokens
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for new_prediction, sample_id in zip(outputs, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})

    all_predictions = predictions

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
    # Prepare LLaVA Model
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.eval()
    model = model.to(device)    

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

    clean_image_processor = lambda x: process_images([x], image_processor, model.config)[0]
    adversarial_image_processor = lambda x: process_images([add_pattern(x, attacker)], image_processor, model.config)[0]
    
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
        zero_shot_clean_result_path = eval_captioning(test_set, model, clean_image_processor, tokenizer, num_shots=0, scaler=scaler, tag='clean')
        
        # Eval Adversarial Performance
        adv_zero_shot_result_path = eval_captioning(test_set, model, adversarial_image_processor, tokenizer, num_shots=0, scaler=scaler, tag='adv')
        
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
        zero_shot_clean_acc = eval_vqa(test_set, model, clean_image_processor, tokenizer, num_shots=0, scaler=scaler, tag='clean')
        # Eval Adversarial Performance
        adv_zero_shot_acc = eval_vqa(test_set, model, adversarial_image_processor, tokenizer, num_shots=0, scaler=scaler, tag='adv')
        
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
    