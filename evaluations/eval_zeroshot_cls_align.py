import argparse
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import time
import open_clip
import torch.nn.functional as F
import XTransferBench
import XTransferBench.zoo
import datasets.utils
import util
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.zero_shot_metadata import zero_shot_meta_dict
from open_clip import get_tokenizer
from tqdm import tqdm
from transformers import AlignProcessor, AlignModel
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
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset for evaluation')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the dataset')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for the evaluation')
# Attacker
parser.add_argument('--threat_model', default='linf_non_targeted', type=str, help='The type of the attacker, see XTransferBench documentation for more details')
parser.add_argument('--attacker_name', default='xtransfer_large_linf_eps12_non_targeted', type=str, help='The name of the attacker, see XTransferBench documentation for more details')

def _convert_to_rgb(image):
    return image.convert('RGB')

def main(args):
     # Prepare Model
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
    model = AlignModel.from_pretrained("kakaobrain/align-base")
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    attacker = XTransferBench.zoo.load_attacker(args.threat_model, args.attacker_name)
    attacker = attacker.to(device)

    # Prepare Data
    data_transforms = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        _convert_to_rgb,
        transforms.ToTensor(),
    ]
    data_transforms = transforms.Compose(data_transforms)
    test_set = datasets.utils.dataset_options[args.dataset](args.data_path, transform=data_transforms, is_test=True, kwargs={})
    data_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    # Build template 
    with torch.no_grad():
        classnames = list(zero_shot_meta_dict[args.dataset+'_CLASSNAMES'])
        if attacker.target_text is not None:
            # Add the target text to the classnames
            classnames.append(attacker.target_text)
        templates = zero_shot_meta_dict[args.dataset+'_TEMPLATES']
        use_format = isinstance(templates[0], str)
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = processor(text=texts, return_tensors="pt").to(device)
            class_embeddings = model.get_text_features(**texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    if attacker.target_text is None:
        # Clean images zero shot evaluation
        acc1_meter = util.AverageMeter()
        acc5_meter = util.AverageMeter()
        for images, labels in tqdm(data_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                images = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
                image_features = model.get_image_features(**images).float()
            logits = 100. * image_features @ zeroshot_weights
            acc1, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc1_meter.update(acc1.item(), len(images))
            acc5_meter.update(acc5.item(), len(images))

        payload = "Zero-shot Top-1: {:.4f} Top-5: {:.4f} ".format(acc1_meter.avg, acc5_meter.avg)
        print('\033[33m'+payload+'\033[0m')
        
    # Adversarial images zero shot evaluation
    adv_acc1_meter = util.AverageMeter()
    adv_acc5_meter = util.AverageMeter()
    for images, labels in tqdm(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if attacker.target_text is not None:
            labels = torch.zeros(len(images)).long().to(device) + len(classnames) - 1 # last idx is the target class
        with torch.no_grad():
            adv_images = attacker.attack(images)
            images = processor(images=adv_images, return_tensors="pt", do_rescale=False).to(device)
            image_features = model.get_image_features(**images).float()
        logits = 100. * image_features @ zeroshot_weights
        acc1, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        adv_acc1_meter.update(acc1.item(), len(images))
        adv_acc5_meter.update(acc5.item(), len(images)) 
    
    if attacker.target_text is not None:
        payload = "ASR: {:.4f} Top-5: {:.4f} ".format(adv_acc1_meter.avg, adv_acc5_meter.avg)
        print('\033[33m'+payload+'\033[0m')
    else:
        payload = "Adversarial Zero-shot Top-1: {:.4f} Top-5: {:.4f} ".format(adv_acc1_meter.avg, adv_acc5_meter.avg)
        print('\033[33m'+payload+'\033[0m')

        payload = "ASR: {:.4f} ".format((acc1_meter.avg - adv_acc1_meter.avg)/acc1_meter.avg)
        print('\033[33m'+payload+'\033[0m')
    return

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seed = args.seed
    args.gpu = device

    start = time.time()
    main(args)
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    print(payload)
    