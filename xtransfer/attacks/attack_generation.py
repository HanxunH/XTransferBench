import torch
import torch.nn as nn
import misc
import os
import torch
import torch.nn.functional as F
import mlconfig 
import torch.distributed as dist
import numpy as np
from torch import nn
from open_clip import tokenizer
from torchvision import transforms
from open_clip import get_tokenizer
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')


def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)

class PGDLInfinity(nn.Module):
    def __init__(self, target_text=None, epsilon=16/255, image_size=224, step_size=1/255, eval_mode=False,
                 attack_type='untargeted',):
        super().__init__()
        self.epsilon = epsilon
        self.logit_scale = 1.0
        self.step_size = step_size
        self.eval_mode = eval_mode
        # Prepare target text
        self.target_text = target_text
        self.attack_type = attack_type
        # Prepare delta and mask
        self.delta = torch.FloatTensor(1, 3, image_size, image_size).uniform_(-self.epsilon, self.epsilon).to(device)
        self.delta = misc.broadcast_tensor(self.delta, 0)

    def sync(self):
        self.delta = misc.broadcast_tensor(self.delta, 0)

    def interpolate_epsilon(self, target_esp):
        self.delta = self.delta * target_esp / self.epsilon
        self.epsilon = target_esp
        self.delta = torch.clamp(self.delta, -self.epsilon, self.epsilon)

    @torch.no_grad()
    def attack(self, images):
        delta = self.delta.to(images.device)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if images.shape[2] != delta.shape[2] or images.shape[3] != delta.shape[3]:
            delta = F.interpolate(delta, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
        else:
            delta = delta
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        x_adv = images + delta
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv
    
    def forward(self, model, images, image_normalization, tokenizer):
        self.delta = Variable(self.delta, requires_grad=True)
        delta = self.delta
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        x_adv = images + delta
        x_adv = torch.clamp(x_adv, 0, 1)

        if self.attack_type == 'untargeted':
            # Untargeted attack
            x_adv_features = model.encode_image(image_normalization(x_adv), normalize=True)
            with torch.no_grad():
                x_features = model.encode_image(image_normalization(images), normalize=True)
            cos_loss = (x_adv_features * x_features).sum(dim=1)
            loss = cos_loss.mean()
        elif self.attack_type == 'targeted':
            x_adv_features = model.encode_image(image_normalization(x_adv), normalize=True)
            # Targeted attack
            target_text = tokenizer([self.target_text]).to(device)
            target_text_embedding = model.encode_text(target_text, normalize=True)
            cos_loss = (x_adv_features * target_text_embedding).sum(dim=1)
            loss = - cos_loss.mean()
        elif self.attack_type == 'etu':
            x_adv_features = model.encode_image(image_normalization(x_adv), normalize=False)

             #loss
            criterion = torch.nn.KLDivLoss(reduction='batchmean')

            local_transform = transforms.RandomResizedCrop(224, scale=(0.1, 0.5))
            local_img_transform = transforms.RandomResizedCrop(224, scale=(0.5, 0.8))
            #ScMix    
            patch_images = local_img_transform(images)
            patch_images1 = local_img_transform(images)
            patch_images1 = patch_images1.to(images.device)
            patch_images = patch_images.to(images.device)
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            dp_images = l * patch_images + (1 - l) *patch_images1
            idx = torch.randperm(images.size(0))
            dp_images = 0.8 * dp_images + 0.2 *images[idx]

            patch_uap_noise = local_transform(delta)
            patch_uap_noise = patch_uap_noise.to(device)

            with torch.no_grad():
                image_output_patch = model.encode_image(image_normalization(patch_images), normalize=False)
                image_output_patch1 = model.encode_image(image_normalization(patch_images1), normalize=False)
                image_embed_p = l * image_output_patch + (1 - l) *image_output_patch1
                x_features = model.encode_image(image_normalization(images), normalize=False)

            image_adv1 = torch.clamp(images + patch_uap_noise, 0, 1)
            image_adv2 = torch.clamp(dp_images + patch_uap_noise, 0, 1)
            image_adv1 = image_normalization(image_adv1)
            image_adv2 = image_normalization(image_adv2)
            image_adv_embed1 = model.encode_image(image_adv1, normalize=False)
            image_adv_embed2 = model.encode_image(image_adv2, normalize=False)
                
            loss_kl_image = criterion(x_adv_features.log_softmax(dim=-1), x_features.softmax(dim=-1)) 
            loss_kl_image1 = criterion(image_adv_embed1.log_softmax(dim=-1), x_features.softmax(dim=-1))
            loss_kl_image2 = criterion(image_adv_embed2.log_softmax(dim=-1), x_features.softmax(dim=-1)) + criterion(image_adv_embed2.log_softmax(dim=-1), image_embed_p.softmax(dim=-1))
            
            loss = - loss_kl_image - loss_kl_image1 - loss_kl_image2
            cos_loss = loss_kl_image.detach()

        results = {
            'loss': loss,
            'cos_loss': cos_loss.mean().item(),
        }
        return results
    
    def save(self, path):
        delta = self.delta
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        torch.save(delta, os.path.join(path, 'delta.pth'))
        return
    
    def load(self, path):
        self.delta = torch.load(os.path.join(path, 'delta.pth'))
        return
    

class L2Attack(nn.Module):
    def __init__(self, target_text=None, c=0.01, image_size=224, eval_mode=False, attack_type='untargeted'):
        super().__init__()
        self.c = c
        self.logit_scale = 1.0
        self.eval_mode = eval_mode
        
        # Prepare target text
        self.target_text = target_text
        self.attack_type = attack_type
        # Prepare delta and mask
        delta = torch.FloatTensor(1, 3, image_size, image_size).uniform_(-0.5, 0.5).to(device)
        self.delta_param = nn.Parameter(delta)

    def sync(self):
        pass

    @torch.no_grad()
    def attack(self, images):
        delta = torch.tanh(self.delta_param)
        delta = delta.to(images.device)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if images.shape[2] != delta.shape[2] or images.shape[3] != delta.shape[3]:
            delta = F.interpolate(delta, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
        else:
            delta = delta
        x_adv = images + delta
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv
    
    def forward(self, model, images, image_normalization, tokenizer):
        delta = torch.tanh(self.delta_param).to(images.device)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        x_adv = images + delta
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv_features = model.encode_image(image_normalization(x_adv), normalize=True)

        if self.attack_type == 'untargeted':
            # Untargeted attack
            with torch.no_grad():
                x_features = model.encode_image(image_normalization(images), normalize=True)
            cos_loss = (x_adv_features * x_features).sum(dim=1)
            loss = cos_loss
        elif self.attack_type == 'targeted':
            # Targeted attack
            target_text = tokenizer([self.target_text]).to(device)
            target_text_embedding = model.encode_text(target_text, normalize=True)
            cos_loss = (x_adv_features * target_text_embedding).sum(dim=1)
            loss = - cos_loss
        
        norm = torch.norm(delta, p=2, dim=[1, 2, 3])
        loss = loss + self.c * norm
        loss = loss.mean()

        results = {
            'loss': loss,
            'cos_loss': cos_loss.mean().item(),
            'l2': norm.mean().item()
        }
        return results
    
    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'state_dict.pth'))
        return
    
    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'state_dict.pth'))
        self.load_state_dict(ckpt)
        return


class PatchAttack(nn.Module):
    def __init__(self, target_text=None, alpha=0.0001, beta=70, image_size=224, eval_mode=False, attack_type='untargeted'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.logit_scale = 1.0
        self.eval_mode = eval_mode
        
        # Prepare target text
        self.target_text = target_text
        self.attack_type = attack_type
        # Prepare delta and mask
        delta = torch.FloatTensor(1, 3, image_size, image_size).uniform_(-0.5, 0.5).to(device)
        mask = torch.FloatTensor(1, 3, image_size, image_size).uniform_(-0.5, 0.5).to(device)
        self.mask_param = nn.Parameter(mask)
        self.delta_param = nn.Parameter(delta)

    def sync(self):
        pass

    @torch.no_grad()
    def attack(self, images):
        mask = (torch.tanh(self.mask_param) + 1) / 2
        delta = (torch.tanh(self.delta_param) + 1) / 2
        mask = mask.to(images.device)
        delta = delta.to(images.device)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if images.shape[2] != delta.shape[2] or images.shape[3] != delta.shape[3]:
            delta = F.interpolate(delta, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, (images.shape[2], images.shape[3]), mode='nearest', align_corners=False)
        else:
            delta = delta
        x_adv = delta * mask + (1 - mask) * images
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv
    
    def forward(self, model, images, image_normalization, tokenizer):
        mask = (torch.tanh(self.mask_param) + 1) / 2
        delta = (torch.tanh(self.delta_param) + 1) / 2

        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        x_adv = delta * mask + (1 - mask) * images
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv_features = model.encode_image(image_normalization(x_adv), normalize=True)

        if self.attack_type == 'untargeted':
            # Untargeted attack
            with torch.no_grad():
                x_features = model.encode_image(image_normalization(images), normalize=True)
            cos_loss = (x_adv_features * x_features).sum(dim=1)
            loss = cos_loss
        elif self.attack_type == 'targeted':
            # Targeted attack
            target_text = tokenizer([self.target_text]).to(device)
            target_text_embedding = model.encode_text(target_text, normalize=True)
            cos_loss = (x_adv_features * target_text_embedding).sum(dim=1)
            loss = - cos_loss
        
        l1_norm = torch.norm(mask, p=1, dim=[1, 2, 3])
        tv = total_variation_loss(mask)
        tv += total_variation_loss(delta)
        norm = l1_norm + tv

        loss = cos_loss + self.alpha * norm + self.beta * tv
        loss = loss.mean()

        results = {
            'loss': loss,
            'cos_loss': cos_loss.mean().item(),
            'tv': tv.mean().item(),
            'l1': l1_norm.mean().item()
        }
        return results
    
    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'state_dict.pth'))
        return
    
    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'state_dict.pth'))
        self.load_state_dict(ckpt)
        return

