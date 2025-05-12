import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from huggingface_hub import PyTorchModelHubMixin

def patch_initialization(noise_percentage, patch_type='rectangle'):
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def mask_generation(patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1] - 14 - patch.shape[1]
    y_location = image_size[1] - 14 - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch ,x_location, y_location


class AdvCLIP(nn.Module, PyTorchModelHubMixin, library_name="XTransferBench",):
    def __init__(self, image_size=224, noise_percentage=0.03, checkpoint_path=None, **kwargs):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.uap = nn.Parameter(torch.rand(1, 3, image_size, image_size))
        patch = patch_initialization(noise_percentage)
        mask, applied_patch, x, y = mask_generation(patch)
        applied_patch = torch.from_numpy(applied_patch)
        self.mask = torch.from_numpy(mask).unsqueeze(0)
        self.image_size = image_size
        # Parameters
        self.delta = nn.Parameter(torch.rand(1, 3, image_size, image_size))

    @torch.no_grad()
    def attack(self, images):
        if torch.min(images) < 0 or torch.max(images) > 1:
            raise('pixel values should be between 0 and 1')
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if images.shape[2] != self.uap.shape[2] or images.shape[3] != self.uap.shape[3]:
            uap = F.interpolate(self.uap, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            mask = F.interpolate(self.mask, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            warnings.warn('The size of UAP is not equal to the size of images, interpolate UAP to the size of images may degrade ASR.')
        else:
            uap = self.uap
            mask = self.mask
        uap = uap.to(images.device)
        mask = mask.to(images.device)
        x_adv = (1-mask) * images + mask * uap
        x_adv = torch.clamp(x_adv, 0, 1).type(images.dtype)
        return x_adv
    
    def forward(self, images):
        return self.attack(images)
    
    def load(self):
        if self.checkpoint_path is None:
            raise ValueError("Checkpoint path is not specified.")
        else:
            delta = torch.load(self.checkpoint_path, map_location='cpu')
            if len(delta.shape) == 3:
                delta = delta.unsqueeze(0)
            self.delta.data = delta
        return