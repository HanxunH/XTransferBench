import torch
import torch.nn as nn
import torchvision.transforms as T


class DefaultCollateFunction(nn.Module):
    def __init__(self, **kwargs):
        super(DefaultCollateFunction, self).__init__()

    def forward(self, batch):
        return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])
