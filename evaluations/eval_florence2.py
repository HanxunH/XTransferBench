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
from transformers import AutoProcessor, AutoModelForCausalLM 
from collections import defaultdict
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision import transforms 
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
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
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
    images = []
    for i in range(len(img)):
        img[i] = img[i].resize((224, 224))
        img[i] = transforms.ToTensor()(img[i])
        img[i] = img[i].unsqueeze(0)
        img[i] = attacker.attack(img[i])[0]
        img[i] = transforms.ToPILImage()(img[i])
        images.append(img[i])
    return images

def eval_captioning(test_dataset, model, image_processor, tokenizer,
                    num_shots=0, max_generation_length=20, num_beams=5,
                    min_generation_length=1, scaler=None, tag='clean'):    
    batch_size = args.batch_size
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn
    )

    predictions = defaultdict()
    for batch in tqdm(test_dataloader):
        for i in range(len(batch["image"])):
            batch['image'][i] = _convert_to_rgb(batch['image'][i])
        
        inputs = image_processor(batch["image"], ["<CAPTION>"] * len(batch['image'])).to(device)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    pixel_values=inputs['pixel_values'],
                    max_new_tokens=1024,
                    num_beams=3,
                )

        # Extract only the new gnerated tokens
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        for i in range(len(outputs)):
            outputs[i] = tokenizer.post_process_generation(outputs[i], task='<CAPTION>', image_size=(batch['image'][i].width, batch['image'][i].height))
            outputs[i] = outputs[i]['<CAPTION>'].replace("<pad>", "")
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


def main():
    # Prepare Florence2 Model
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
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

    if args.task == 'vqa':
        raise NotImplementedError("VQA Evaluation is not supported for Florence-2")
    else:
        clean_image_processor = lambda x, prompt: processor(text=prompt, images=x, return_tensors="pt").to(device, torch_dtype)
        adversarial_image_processor = lambda x, prompt: processor(text=prompt, images=add_pattern(x, attacker), return_tensors="pt").to(device, torch_dtype)
    tokenizer = processor

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
        raise NotImplementedError("VQA Evaluation is not supported for Florence-2")
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
    