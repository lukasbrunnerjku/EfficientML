import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .unet import UNet


class KeypointUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 512,
        n_keypoints: int = 5,
        upscale_factor: int = 4,
        offset_channels: int = 2,
        aux_channels: int = 1,
    ):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.upscale_factor = upscale_factor
        self.offset_channels = offset_channels
        self.aux_channels = aux_channels
        self.n_head_channels = (
            n_keypoints + offset_channels + aux_channels
        ) * upscale_factor**2  # semantic heatmaps, offsets in x,y folded in ie. 4x4 superpixels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.n_head_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_head_channels, self.n_head_channels, 3, 1, 1, bias=True),
        )
        self.layers[-1].bias.data.fill_(-2.19)

    def forward(self, x: Tensor):
        cells = self.layers(x)  # ie. Bx(K+2+1)*r*rxH''xW''
        if self.upscale_factor > 1:
            cells = F.pixel_shuffle(cells, self.upscale_factor)
        heatmaps = cells[:, : self.n_keypoints]  # BxKxH'xW'
        offsets = cells[
            :, self.n_keypoints : self.n_keypoints + self.offset_channels
        ]  # Bx2xH'xW'
        offsets = offsets.permute(0, 2, 3, 1)  # BxH'xW'x2
        aux = cells[:, self.n_keypoints + self.offset_channels :]  # Bx1xH'xW'
        return {"heatmaps": heatmaps, "offsets": offsets, "aux": aux}


if __name__ == "__main__":
    model = UNet(3, (64, 128, 256), (2, 2, 3), with_stem=True, add_residual=False)
    model.conv_out = KeypointUpsampleBlock(
        model.block_out_channels[0],
        hidden_channels=4 * model.block_out_channels[0],
        upscale_factor=4,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet with {1e-6 * n_params:.3f} M parameters.")
    
    model.eval()
    model.cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    
    output = model(x)
    for k, v in output.items():
        print(f"{k}: {v.shape}")
    