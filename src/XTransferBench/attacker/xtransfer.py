import torch
import torch.nn as nn
import os
import torch
import torch.nn.functional as F
import warnings
from huggingface_hub import PyTorchModelHubMixin


class XTransferLinf(nn.Module, PyTorchModelHubMixin, library_name="XTransferBench"):
    def __init__(self, epsilon=12/255, image_size=224, checkpoint_path=None, target_text=None, **kwargs):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.epsilon = epsilon
        self.target_text = target_text
        self.image_size = image_size
        # Parameters
        self.delta = nn.Parameter(torch.rand(1, 3, image_size, image_size))
        
    def interpolate_epsilon(self, target_esp):
        self.delta.data = self.delta.data * target_esp / self.epsilon
        self.epsilon = target_esp
        self.delta.data = torch.clamp(self.delta.data, -self.epsilon, self.epsilon)

    @torch.no_grad()
    def attack(self, images):
        if torch.min(images) < 0 or torch.max(images) > 1:
            raise('pixel values should be between 0 and 1')
        delta = self.delta.to(images.device)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if images.shape[2] != delta.shape[2] or images.shape[3] != delta.shape[3]:
            delta = F.interpolate(delta, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            warnings.warn('The size of UAP is not equal to the size of images, interpolate UAP to the size of images may degrade ASR.')
        else:
            delta = delta
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        x_adv = images + delta
        x_adv = torch.clamp(x_adv, 0, 1)
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
    

class XTransferL2Attack(nn.Module, PyTorchModelHubMixin, library_name="XTransferBench"):
    def __init__(self, image_size=224, checkpoint_path=None, target_text=None, **kwargs):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.target_text = target_text
        self.image_size = image_size
        # Parameters
        delta = torch.FloatTensor(1, 3, image_size, image_size).uniform_(-0.5, 0.5)
        self.delta_param = nn.Parameter(delta)

    @torch.no_grad()
    def attack(self, images):
        if torch.min(images) < 0 or torch.max(images) > 1:
            raise('pixel values should be between 0 and 1')
        delta = torch.tanh(self.delta_param)
        delta = delta.to(images.device)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if images.shape[2] != delta.shape[2] or images.shape[3] != delta.shape[3]:
            delta = F.interpolate(delta, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            warnings.warn('The size of UAP is not equal to the size of images, interpolate UAP to the size of images may degrade ASR.')
        else:
            delta = delta
        x_adv = images + delta
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv
    
    def forward(self, images):
        return self.attack(images)
    
    def load(self):
        if self.checkpoint_path is None:
            raise ValueError("Checkpoint path is not specified.")
        else:
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')
            self.load_state_dict(ckpt)
        return
    
