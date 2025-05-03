import mlconfig
import torch
from .attack_generation import PGDLInfinity, L2Attack, PatchAttack


mlconfig.register(PGDLInfinity)
mlconfig.register(L2Attack)
mlconfig.register(PatchAttack)

mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.AdamW)
mlconfig.register(torch.optim.LBFGS)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
