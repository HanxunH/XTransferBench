import argparse
import torch
import torch.nn as nn
import mlconfig
import datasets
import attacks
import util
import misc
import os
import sys
import numpy as np
import time
import open_clip
import torch.nn.functional as F
from torchvision import transforms
from exp_mgmt import ExperimentManager
from datasets.zero_shot_metadata import zero_shot_meta_dict
from open_clip import get_tokenizer
mlconfig.register(open_clip.create_model_and_transforms)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='XTransfer')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)

def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    
    # Prepare Model
    model, _, preprocess = config.model()
    _normalize = preprocess.transforms[-1]
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if hasattr(exp.config, 'overload_ckpt'):
        ckpt_path = exp.config.overload_ckpt
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            name = k[7:] # remove `module.`
            if name.startswith('encode_text.'):
                name = name.replace('encode_text', '')
            new_state_dict[name] = v
        
        print(new_state_dict.keys())
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)

    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None

    attacker = config.attacker(eval_mode=True)
    target_path = config.attacker_checkpoint_path
    attacker.load(target_path)
    results = {}

    # Prepare Data
    config.dataset.train_tf = preprocess.transforms[:-1]
    config.dataset.test_tf = preprocess.transforms[:-1]
    data = config.dataset()
    loader = data.get_loader(drop_last=False)
    _, test_loader, _ = loader

    # Zero shot evaluation
    # Build template 
    with torch.no_grad():
        classnames = list(zero_shot_meta_dict[config.class_names])
        templates = zero_shot_meta_dict[config.zero_shot_templates]
        use_format = isinstance(templates[0], str)
        zeroshot_weights = []
        if (exp.config.model['model_name'], exp.config.model['pretrained']) in open_clip.list_pretrained() or "hf-hub" in exp.config.model['model_name']:
            clip_tokenizer = get_tokenizer(exp.config.model.model_name) if hasattr(exp.config.model, 'model_name') else None
        else:
            clip_tokenizer = None
        for classname in classnames:
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = clip_tokenizer(texts).to(device) if clip_tokenizer is not None else texts
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                class_embeddings = model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    acc1_meter = util.AverageMeter()
    acc5_meter = util.AverageMeter()
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            if config.normalize_image:
                images = _normalize(images)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                image_features = model.encode_image(images, normalize=True)
        logits = 100. * image_features @ zeroshot_weights
        acc1, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        acc1_meter.update(acc1.item(), len(images))
        acc5_meter.update(acc5.item(), len(images))

    results['clean_test_acc1'] = acc1_meter.avg
    results['clean_test_acc5'] = acc5_meter.avg

    payload = "Zero-shot Top-1: {:.4f} Top-5: {:.4f} ".format(acc1_meter.avg, acc5_meter.avg)
    logger.info('\033[33m'+payload+'\033[0m')
        
    # Attack ASR zero shot evaluation
    # Build template 
    if hasattr(exp.config, 'target_text') and exp.config.target_text is not None:
        with torch.no_grad():
            classnames = list(zero_shot_meta_dict[config.class_names])
            classnames.append(exp.config.target_text)
            templates = zero_shot_meta_dict[config.zero_shot_templates]
            use_format = isinstance(templates[0], str)
            zeroshot_weights = []
            if (exp.config.model['model_name'], exp.config.model['pretrained']) in open_clip.list_pretrained() or "hf-hub" in exp.config.model['model_name']:
                clip_tokenizer = get_tokenizer(exp.config.model.model_name) if hasattr(exp.config.model, 'model_name') else None
            else:
                clip_tokenizer = None
            for classname in classnames:
                texts = [template.format(classname) if use_format else template(classname) for template in templates]
                texts = clip_tokenizer(texts).to(device) if clip_tokenizer is not None else texts
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    class_embeddings = model.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    targeted_acc1_meter = util.AverageMeter()
    targeted_acc5_meter = util.AverageMeter()
    untarget_acc1_meter = util.AverageMeter()
    untarget_acc5_meter = util.AverageMeter()
    for images, truelabels in test_loader:
        images = images.to(device, non_blocking=True)
        truelabels = truelabels.to(device, non_blocking=True)
        labels = torch.zeros(len(images)).long().to(device) + len(classnames) - 1 # last idx is the target class
        images = attacker.attack(images)
        with torch.no_grad():
            if config.normalize_image:
                images = _normalize(images)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                image_features = model.encode_image(images, normalize=True)
        logits = 100. * image_features @ zeroshot_weights

        if hasattr(exp.config, 'target_text') and exp.config.target_text is not None:
            targeted_acc1, targeted_acc5 = util.accuracy(logits, labels, topk=(1, 5))
            targeted_acc1_meter.update(targeted_acc1.item(), len(images))
            targeted_acc5_meter.update(targeted_acc5.item(), len(images))

        untarget_acc1, untarget_acc5 = util.accuracy(logits, truelabels, topk=(1, 5))
        untarget_acc1_meter.update(untarget_acc1.item(), len(images))
        untarget_acc5_meter.update(untarget_acc5.item(), len(images))
    
    if hasattr(exp.config, 'target_text') and exp.config.target_text is not None:
        results['attack_test_targeted_acc1'] = targeted_acc1_meter.avg
        results['attack_test_targeted_acc5'] = targeted_acc5_meter.avg
    results['attack_test_untarget_acc1'] = untarget_acc1_meter.avg
    results['attack_test_untarget_acc5'] = untarget_acc5_meter.avg
    results['attack_test_relative_asr1'] = (acc1_meter.avg - untarget_acc1_meter.avg) / acc1_meter.avg
    results['attack_test_relative_asr5'] = (acc5_meter.avg - untarget_acc5_meter.avg) / acc5_meter.avg

    if hasattr(exp.config, 'target_text') and exp.config.target_text is not None:
        payload = "Attack Zero-shot Targeted Top-1: {:.4f} Top-5: {:.4f} ".format(targeted_acc1_meter.avg, targeted_acc5_meter.avg)
        logger.info('\033[33m'+payload+'\033[0m')
    payload = "Attack Zero-shot Untarget Top-1: {:.4f} Top-5: {:.4f} ".format(untarget_acc1_meter.avg, untarget_acc5_meter.avg)
    logger.info('\033[33m'+payload+'\033[0m')
    payload = "Attack Zero-shot Relative ASR Top-1: {:.4f} Top-5: {:.4f} ".format(results['attack_test_relative_asr1'], results['attack_test_relative_asr5'])
    logger.info('\033[33m'+payload+'\033[0m')
    
    # Save results
    exp.save_eval_stats(results, 'zero_shot_eval')
    
    return

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seed = args.seed
    args.gpu = device

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(
        args=args,
        exp_name=args.exp_name,
        exp_path=args.exp_path,
        config_file_path=config_filename
    )
    experiment.config.dataset.seed = args.seed
    
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    