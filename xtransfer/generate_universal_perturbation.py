import argparse
import torch
import torch.nn as nn
import mlconfig
import datasets
import attacks
import search
import util
import misc
import os
import sys
import numpy as np
import time
import open_clip
import itertools
import wandb
import torch.nn.functional as F
from exp_mgmt import ExperimentManager
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
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
parser.add_argument('--load_model', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

def init_models(model_index, model_list):
    # Surrogate Model Initialization
    model_args = model_list[model_index]
    if 'huggingface' in model_args['model_name']:
        model = CLIPModel.from_pretrained(model_args['model_name'].replace('huggingface:', ''))
        processor = CLIPProcessor.from_pretrained(model_args['model_name'].replace('huggingface:', ''))
        tokenizer = processor.tokenizer
        _normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        model_name = model_args['model_name'].replace('huggingface:', '')
        def encode_image(x, normalize):
            x = model.get_image_features(x)
            if normalize:
                x = F.normalize(x, dim=-1)
            return x
        def encode_text(x, normalize):
            x = model.get_text_features(x)
            if normalize:
                x = F.normalize(x, dim=-1)
            return x
        model.encode_image = encode_image
        model.encode_text = encode_text
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(**model_args)
        model_name = model_args['model_name']
        tokenizer = open_clip.get_tokenizer(model_name)
        _normalize = preprocess.transforms[-1]
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, model_name, tokenizer, _normalize

def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config

    # Prepare Model
    model_list = config.model_list # Ensemble list
    if hasattr(config, 'searcher'):
        searcher = config.searcher(len(model_list))
    else:
        searcher = None

    # Prepare Data
    data = config.dataset()
    sampler_train = torch.utils.data.RandomSampler(data.train_set)
    sampler_val = torch.utils.data.SequentialSampler(data.test_set)
    loader = data.get_loader(drop_last=True, train_shuffle=True, train_sampler=sampler_train, test_sampler=sampler_val)
    train_loader, _, _ = loader

    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None

    global_step = 0
    start_epoch = 0
    attacker = config.attacker()

    if 'PGD' in exp.config.attacker.name:
        optimizer = None
    else:
        optimizer = config.optimizer(attacker.parameters())

    if args.ddp and 'PGD' not in exp.config.attacker.name:
        attacker = torch.nn.parallel.DistributedDataParallel(attacker, device_ids=[args.gpu], find_unused_parameters=False)
        attacker_without_ddp = attacker.module
    else:
        attacker_without_ddp = attacker

    # Surrogate Model Initialization
    if searcher is not None:
        model_index = np.random.choice(range(len(model_list)))
    else:
        model_index = misc.get_rank()
    model, model_name, tokenizer, _normalize = init_models(model_index, model_list)
    
    # Generate Universal Perturbation
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=exp.config.log_frequency, fmt='{value:.4f}'))
    all_stats_keys = None
    for epoch in range(start_epoch, exp.config.epochs):
        torch.cuda.synchronize()
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        for i, (images, texts) in enumerate(train_loader):
            if hasattr(exp.config, 'lr_schedule') and optimizer is not None:
                util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
            start = time.time()
            
            if searcher is not None and global_step % exp.config.search_frequency == 0:
                # Update model
                tok_k_index = searcher(misc.get_world_size()) 
                model_index = tok_k_index[misc.get_rank()] 
                model, model_name, tokenizer, _normalize = init_models(model_index, model_list)
                
            # To Support A100 80G 
            if 'bigG-14' in model_name or 'E-14-plus' in model_name:
                if hasattr(attacker, 'attack_type') and attacker.attack_type == 'etu':
                    images = images[:8]
                else:
                    images = images[:32]
            elif 'ViT-H' in model_name or 'ViTamin-L' in model_name or 'g-14' in model_name:
                if hasattr(attacker, 'attack_type') and attacker.attack_type == 'etu':
                    images = images[:16]
                else:
                    images = images[:64]
            elif 'L-14' in model_name or 'large_d' in model_name:
                if hasattr(attacker, 'attack_type') and attacker.attack_type == 'etu':
                    images = images[:32]
                else:
                    images = images[:128]
        
            images = images.cuda(non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad()
                
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                results = attacker(model, images, _normalize, tokenizer)
                loss = results['loss']

            if scaler is not None:
                scaler.scale(loss).backward()
                if optimizer is not None:
                    scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if optimizer is not None:   
                    optimizer.step()

            # PGD
            if optimizer is None:
                grad_sign = attacker.delta.grad.clone().sign()
                grad_sign = misc.all_reduce_sum(grad_sign) / misc.get_world_size()
                attacker.delta = attacker.delta.clone() - exp.config.attacker.step_size * grad_sign
                attacker.delta = torch.clamp(attacker.delta, -exp.config.attacker.epsilon, exp.config.attacker.epsilon)
                attacker.sync() 
                
            # Update searcher
            if searcher is not None:
                if hasattr(exp.config, 'reward_metric'):
                    reward = results[exp.config.reward_metric]
                else:
                    reward = loss.detach().clone()
                if misc.get_world_size() > 1:
                    full_rank_model_index = torch.cat(misc.gather(torch.tensor([model_index], device=device)), dim=0)
                    full_rank_loss = torch.cat(misc.gather(torch.tensor([reward], device=device)), dim=0)
                else:
                    full_rank_model_index = [model_index]
                    full_rank_loss = [torch.tensor(reward)]
            
                for rank, item in enumerate(full_rank_loss):
                    searcher.update(full_rank_model_index[rank].item(), item.item())

            # Update Meters
            batch_size = images.shape[0]

            if all_stats_keys is None:
                stats_keys = list(results.keys())
                all_stats_keys = misc.all_gather(stats_keys)
                all_stats_keys = itertools.chain(*all_stats_keys)
                all_stats_keys = list(set(all_stats_keys))
                all_stats_keys.sort()
                for k in all_stats_keys:
                    metric_logger.add_meter(k, misc.SmoothedValue(window_size=exp.config.log_frequency, fmt='{value:.4f}'))
                metric_logger.synchronize_between_processes()

            metric_logger.update(loss=loss.item(), n=batch_size)            
            for k in all_stats_keys:
                if k == 'batch_size' or k == 'epsilon':
                    continue
                if k in results:
                    v = results[k]
                    metric_logger.update(**{k: v}, n=batch_size)

            # Log results
            end = time.time()
            time_used = end - start
            if optimizer is not None:
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = attacker.module.step_size if hasattr(attacker, 'module') else attacker.step_size
            
            if global_step % exp.config.log_frequency == 0:
                metric_logger.synchronize_between_processes()
                log_payload = {
                    "lr": lr,
                }
                for k in all_stats_keys:
                    if k == 'batch_size':
                        continue
                    if 'rank' in k:
                        pass
                    else:
                        log_payload[k] = metric_logger.meters[k].global_avg
                if misc.get_rank() == 0:
                    display = util.log_display(epoch=epoch, global_step=global_step, time_elapse=time_used, **log_payload)
                    logger.info(display)                    
                metric_logger.reset()

            global_step += 1

        metric_logger.reset()

        # Save model
        if misc.get_rank() == 0:
            # Save model
            attacker_without_ddp.save(exp.checkpoint_path)
            if searcher is not None:
                exp.save_epoch_stats(epoch, searcher.get_stats())
            logger.info('Model saved')

    return

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
        seed = args.seed
    args.gpu = device

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(
        args=args, exp_name=args.exp_name, exp_path=args.exp_path,
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
    if args.ddp: 
        misc.destroy_process_group()