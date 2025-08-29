import torch
from torch.nn import functional

import sys
REPO_DIR = '/mnt/sandbox1/li.yu/code/dinov3'
sys.path.append(REPO_DIR)

class DINOv3VIT7B(torch.nn.Module):
    """BRT Segmentation model with definition to make it a custom model supported."""
    def __init__(self) -> None:
        super().__init__()

        # define backbone
        self.backbone = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', pretrained=False)

    def forward(self, x):
        y = self.backbone(x)
        return y  # Nx4096
