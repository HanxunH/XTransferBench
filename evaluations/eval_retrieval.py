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
import random
import copy
from torchvision import transforms
from torch.utils.data import DataLoader
from open_clip import get_tokenizer
from tqdm import tqdm

def _convert_to_rgb(image):
    return image.convert('RGB')

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
parser.add_argument('--dataset', default='re_eval_val_dataset', type=str, help='Dataset for evaluation')
parser.add_argument('--data_path', default='/path/to/coco2014', type=str, help='Path to the dataset')
parser.add_argument('--val_annotations_path', default='/path/to/coco_test.json', type=str, help='Path to the annotations file')
parser.add_argument('--image_val_dir_path', default='/path/to/coco2014', type=str, help='Path to the image directory')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for the evaluation')
# Victim Model
parser.add_argument('--model', default='ViT-L-14', type=str, help='CLIP encoder evaluation model, See OpenCLIP documentation for more details')
parser.add_argument('--pretrained', default='openai', type=str, help='See OpenCLIP documentation for more details')
parser.add_argument('--cache_dir', default=None, type=str, help='Cache directory checkpoints')
# Attacker
parser.add_argument('--threat_model', default='linf_non_targeted', type=str, help='The type of the attacker, see XTransferBench documentation for more details')
parser.add_argument('--attacker_name', default='xtransfer_large_linf_eps12_non_targeted', type=str, help='The name of the attacker, see XTransferBench documentation for more details')


class DefaultCollateFunction(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DefaultCollateFunction, self).__init__()

    def forward(self, batch):
        return [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, img2txt_only=False, txt2img_only=False):
    
    #Images->Text 
    if not txt2img_only:
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        tr_mean = (tr1 + tr5 + tr10) / 3
        tr_ranks = ranks
    else:
        tr1, tr5, tr10, tr_mean = None, None, None, None
        tr_ranks = None

    if img2txt_only:
        return {'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': tr_mean}, tr_ranks, None
    
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # print(inds, txt2img[index])
        ranks[index] = np.where(inds == txt2img[index])[0][0]
    ir_ranks = ranks

    # Compute metrics

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        
    ir_mean = (ir1 + ir5 + ir10) / 3
    if txt2img_only:
        return {'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': ir_mean}, None, ir_ranks

    r_mean = (tr_mean + ir_mean) / 2
    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result, tr_ranks, ir_ranks

def _convert_to_rgb(image):
    return image.convert('RGB')

def add_pattern(img, attacker):
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = attacker.attack(img)[0]
    img = transforms.ToPILImage()(img)
    return img

def main(args):
    # Prepare Model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, args.pretrained, cache_dir=args.cache_dir)
    _normalize = preprocess.transforms[-1]
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
    test_set = datasets.utils.dataset_options[args.dataset](args.data_path, transform=data_transforms, is_test=True, kwargs={
        "val_annotations_path": args.val_annotations_path,
        "image_val_dir_path": args.image_val_dir_path,
    })
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=DefaultCollateFunction())
    clip_tokenizer = get_tokenizer(args.model)
    
    # Build template 
    texts = test_set.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []  
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = clip_tokenizer(text).to(device) 
        text_embed = model.encode_text(text_input)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
        text_embeds.append(text_embed) 
    text_embeds = torch.cat(text_embeds,dim=0)
    
    image_embeds = []
    with torch.inference_mode():
        for batch in tqdm(test_loader): 
            image = batch[0]
            image_tensor = []
            for i in range(len(image)):
                image_tensor.append(preprocess(_convert_to_rgb(image[i])))
            image = torch.stack(image_tensor, dim=0)
            image = image.to(device) 
            image_embed = model.encode_image(image)        
            image_embed /= image_embed.norm(dim=-1, keepdim=True)     
            image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)
    sims_matrix = image_embeds @ text_embeds.t()
    score_i2t, score_t2i = sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()
    clean_results, clean_tr_ranks, clean_ir_ranks = itm_eval(score_i2t, score_t2i, test_set.txt2img, test_set.img2txt)
    print('\033[33m'+"Clean Evaluation"+'\033[0m')
    payload = "TR@1: {:.2f} TR@5: {:.2f} TR@10: {:.2f} TR_Avg: {:.2f}".format(clean_results['txt_r1'], clean_results['txt_r5'], clean_results['txt_r10'], clean_results['txt_r_mean'])
    print('\033[33m'+payload+'\033[0m')
    payload = "IR@1: {:.2f} IR@5: {:.2f} IR@10: {:.2f} IR: {:.2f}".format(clean_results['img_r1'], clean_results['img_r5'], clean_results['img_r10'], clean_results['img_r_mean'])
    print('\033[33m'+payload+'\033[0m')


    ###########################################################################
    # Untargeted Adversarial Evaluation Text Retrieval
    # add noise/patch to all images
    image_embeds = []
    with torch.inference_mode():
        for batch in tqdm(test_loader): 
            image = batch[0]
            image_tensor = []
            for i in range(len(image)):
                image_tensor.append(preprocess(add_pattern(_convert_to_rgb(image[i]), attacker)))
            image = torch.stack(image_tensor, dim=0)
            image = image.to(device) 
            image_embed = model.encode_image(image)        
            image_embed /= image_embed.norm(dim=-1, keepdim=True)     
            image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)
    sims_matrix = image_embeds @ text_embeds.t()
    score_i2t, score_t2i = sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()
    untargeted_adv_results, untargeted_adv_tr_ranks, untargeted_adv_ir_ranks = itm_eval(score_i2t, score_t2i, test_set.txt2img, test_set.img2txt)
    
    print('\033[33m'+"Untargeted Adv Evaluation"+'\033[0m')
    payload = "TR@1: {:.2f} TR@5: {:.2f} TR@10: {:.2f} TR_Avg: {:.2f}".format(untargeted_adv_results['txt_r1'], untargeted_adv_results['txt_r5'], untargeted_adv_results['txt_r10'], clean_results['txt_r_mean'])
    print('\033[33m'+payload+'\033[0m')
    payload = "IR@1: {:.2f} IR@5: {:.2f} IR@10: {:.2f} IR: {:.2f}".format(untargeted_adv_results['img_r1'], untargeted_adv_results['img_r5'], untargeted_adv_results['img_r10'], untargeted_adv_results['img_r_mean'])
    print('\033[33m'+payload+'\033[0m')
    payload = "Relative ASRTR@1: {:.2f} Relative ASRTR@5: {:.2f} Relative ASRTR@10: {:.2f} Relative TRASR: {:.2f}".format(
        (clean_results['txt_r1'] - untargeted_adv_results['txt_r1']) / clean_results['txt_r1'] * 100, 
        (clean_results['txt_r5'] - untargeted_adv_results['txt_r5']) / clean_results['txt_r5'] * 100, 
        (clean_results['txt_r10'] - untargeted_adv_results['txt_r10']) / clean_results['txt_r10'] * 100, 
        (clean_results['txt_r_mean'] - untargeted_adv_results['txt_r_mean']) / clean_results['txt_r_mean'] * 100)
    print('\033[33m'+payload+'\033[0m')
    payload = "Relative ASRIR@1: {:.2f} Relative ASRIR@5: {:.2f} Relative ASRIR@10: {:.2f} Relative AIR: {:.2f}".format(
        (clean_results['img_r1'] - untargeted_adv_results['img_r1']) / clean_results['img_r1'] * 100, 
        (clean_results['img_r5'] - untargeted_adv_results['img_r5']) / clean_results['img_r5'] * 100, 
        (clean_results['img_r10'] - untargeted_adv_results['img_r10']) / clean_results['img_r10'] * 100, 
        (clean_results['img_r_mean'] - untargeted_adv_results['img_r_mean']) / clean_results['img_r_mean'] * 100)
    print('\033[33m'+payload+'\033[0m')

    # Set diff ASR
    clean_tr1 = np.where(clean_tr_ranks < 1)[0]
    clean_tr5 = np.where(clean_tr_ranks < 5)[0]
    clean_tr10 = np.where(clean_tr_ranks < 10)[0]
    clean_ir1 = np.where(clean_ir_ranks < 1)[0]
    clean_ir5 = np.where(clean_ir_ranks < 5)[0]
    clean_ir10 = np.where(clean_ir_ranks < 10)[0]

    untargeted_adv_tr1 = np.where(untargeted_adv_tr_ranks < 1)[0]
    untargeted_adv_tr5 = np.where(untargeted_adv_tr_ranks < 5)[0]
    untargeted_adv_tr10 = np.where(untargeted_adv_tr_ranks < 10)[0]
    untargeted_adv_ir1 = np.where(untargeted_adv_ir_ranks < 1)[0]
    untargeted_adv_ir5 = np.where(untargeted_adv_ir_ranks < 5)[0]
    untargeted_adv_ir10 = np.where(untargeted_adv_ir_ranks < 10)[0]
    
    set_diff_asr_tr1 = round(100.0 * len(np.setdiff1d(clean_tr1, untargeted_adv_tr1)) / len(clean_tr1), 2) 
    set_diff_asr_tr5 = round(100.0 * len(np.setdiff1d(clean_tr5, untargeted_adv_tr5)) / len(clean_tr5), 2)
    set_diff_asr_tr10 = round(100.0 * len(np.setdiff1d(clean_tr10, untargeted_adv_tr10)) / len(clean_tr10), 2)

    set_diff_asr_ir1 = round(100.0 * len(np.setdiff1d(clean_ir1, untargeted_adv_ir1)) / len(clean_ir1), 2)
    set_diff_asr_ir5 = round(100.0 * len(np.setdiff1d(clean_ir5, untargeted_adv_ir5)) / len(clean_ir5), 2)
    set_diff_asr_ir10 = round(100.0 * len(np.setdiff1d(clean_ir10, untargeted_adv_ir10)) / len(clean_ir10), 2)
    
    payload = "Set Diff ASRTR@1: {:.2f} Set Diff ASRTR@5: {:.2f} Set Diff ASRTR@10: {:.2f}".format(
        set_diff_asr_tr1, set_diff_asr_tr5, set_diff_asr_tr10)
    print('\033[33m'+payload+'\033[0m')
    payload = "Set Diff ASRIR@1: {:.2f} Set Diff ASRIR@5: {:.2f} Set Diff ASRIR@10: {:.2f}".format(
        set_diff_asr_ir1, set_diff_asr_ir5, set_diff_asr_ir10)
    print('\033[33m'+payload+'\033[0m')
    
    ###########################################################################
    # Targeted Adversarial Evaluation Text Retrieval
    # Add one more text (the target text)
    if hasattr(attacker, 'target_text') and attacker.target_text is not None:
        text_embeds = []  
        texts = test_set.text   
        texts.append(attacker.target_text)
        num_text = len(texts)
        text_bs = args.batch_size
        text_embeds = []  
        for i in tqdm(range(0, num_text, text_bs)):
            text = texts[i: min(num_text, i+text_bs)]
            text_input = clip_tokenizer(text).to(device) 
            text_embed = model.encode_text(text_input)
            text_embed /= text_embed.norm(dim=-1, keepdim=True)
            text_embeds.append(text_embed) 
        text_embeds = torch.cat(text_embeds,dim=0)
        # Add noise/patch to all images
        image_embeds = []
        with torch.inference_mode():
            for batch in tqdm(test_loader): 
                image = batch[0]
                image_tensor = []
                for i in range(len(image)):
                    image_tensor.append(preprocess(add_pattern(_convert_to_rgb(image[i]), attacker)))
                image = torch.stack(image_tensor, dim=0)
                image = image.to(device) 
                image_embed = model.encode_image(image)        
                image_embed /= image_embed.norm(dim=-1, keepdim=True)     
                image_embeds.append(image_embed)
        image_embeds = torch.cat(image_embeds,dim=0)
        sims_matrix = image_embeds @ text_embeds.t()
        score_i2t, score_t2i = sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()
        targeted_img2txt = copy.deepcopy(test_set.img2txt)
        for key in targeted_img2txt:
            targeted_img2txt[key] = [num_text-1]
        adv_tr_results, _, _ = itm_eval(score_i2t, score_t2i, None, targeted_img2txt, img2txt_only=True)
        print('\033[33m'+"Targeted Adv Evaluation"+'\033[0m')
        payload = "TR@1: {:.2f} TR@5: {:.2f} TR@10: {:.2f} TR_Avg: {:.2f}".format(adv_tr_results['txt_r1'], adv_tr_results['txt_r5'], adv_tr_results['txt_r10'], adv_tr_results['txt_r_mean'])
        print('\033[33m'+payload+'\033[0m')
        

        ###########################################################################
        # Targeted Adversarial Evaluation Image Retrieval
        # obtain target text_embedding
        text_input = clip_tokenizer([attacker.target_text]).to(device) 
        text_embeds = model.encode_text(text_input)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        # Add trigger to first image only
        rank_list = []
        for _ in tqdm(range(50)):
            target_idx = random.randint(0, len(test_set)-1)
            image_embeds = []
            global_idx = 0
            with torch.inference_mode():
                for batch in test_loader: 
                    image = batch[0]
                    image_tensor = []
                    for i in range(len(image)):
                        if global_idx==target_idx:
                            image_tensor.append(preprocess(add_pattern(_convert_to_rgb(image[i]), attacker)))
                        else:
                            image_tensor.append(preprocess(_convert_to_rgb(image[i])))
                        global_idx += 1
                    image = torch.stack(image_tensor, dim=0)
                    image = image.to(device) 
                    image_embed = model.encode_image(image)        
                    image_embed /= image_embed.norm(dim=-1, keepdim=True)     
                    image_embeds.append(image_embed)
            image_embeds = torch.cat(image_embeds, dim=0)
            sims_matrix = image_embeds @ text_embeds.t()
            score_t2i = sims_matrix.t().cpu().numpy()
            inds = np.argsort(score_t2i[0])[::-1]
            rank = int(np.where(inds == target_idx)[0][0])
            rank_list.append(rank)

        targeted_adv_ir_results = {
            'targeted_adv_rank_avg': sum(rank_list) / len(rank_list),
            'targeted_adv_rank_std': float(np.std(rank_list)),
            'targeted_adv_rank_list': rank_list,
            'targeted_adv_full_rank': len(inds)
        }
        payload = "Average Rank: {:.4f} Average Rank Std: {:.4f}".format(targeted_adv_ir_results['targeted_adv_rank_avg'], targeted_adv_ir_results['targeted_adv_rank_std'])
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
    