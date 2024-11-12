import torch
from torch.nn import functional

import sys
sys.path.append('/home/li.yu/code/JupiterCVML/europa/base/src/europa')
from dl.network.nextvit_brt import _get_nextvit

class NextVitSmall(torch.nn.Module):
    """BRT Segmentation model with definition to make it a custom model supported."""
    def __init__(self, num_classes) -> None:
        super().__init__()

        # define backbone
        self.backbone = _get_nextvit(
            model_size="small",
            frozen_stages=-1,
            norm_eval=False,
            with_extra_norm=True,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            in_channels=3,
        )

        # self.proj_head = torch.nn.Sequential(
        #     torch.nn.Linear(1024, num_classes),
        # )
        self.proj_head = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = functional.adaptive_avg_pool2d(y[-1], (1, 1))
        y = torch.flatten(y, 1)
        y = self.proj_head(y)

        return y


def load_weights(model):
    # load checkpoint
    ckpt_path = '/data/jupiter/li.yu/exps/driveable_terrain_model/openimages_v7_0131/checkpoint_brt_compatible.pth'
    # ckpt_path = '/data/jupiter/li.yu/exps/driveable_terrain_model/lightly_densecldino_0927/checkpoints/last.pth'
    # ckpt_path = '/mnt/sandbox1/ben.cline/logs/bc_sandbox_2024_q4/11_1_rev1_train_human_test_dean_multires/checkpoints/best.ckpt'
    input_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    if len(input_dict.keys()) < 10:  # when contain other states like from optimizer, epoch number etc
        print(input_dict.keys())

    if 'model' in input_dict:
        state_dict = input_dict['model']
    elif 'state_dict' in input_dict:
        state_dict = input_dict['state_dict']
    else:
        state_dict = input_dict
    if 'model.backbone.stem.0.conv.weight' in state_dict:
        new_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith('model.backbone'):
                new_state_dict[k[6:]] = v
        state_dict = new_state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f'# missing keys {len(missing_keys)}, # unexpected keys {len(unexpected_keys)}')
    print(f'# model keys {len(model.state_dict())}, # loaded keys {len(state_dict)}')
    print('tunable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('loaded weights from', ckpt_path)
    return model