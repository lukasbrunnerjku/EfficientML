import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm.layers import trunc_normal_, DropPath

from .utils import AdaLayerNorm, CondSequential


class StemLayer(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        embed_dim: Optional[int] = None,
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.norm1 = AdaLayerNorm(out_channels // 2, embed_dim, num_embeddings, adaLN_Zero=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.norm2 = AdaLayerNorm(out_channels, embed_dim, num_embeddings, adaLN_Zero=False)

    def forward(
        self,
        x: Tensor,
        emb: Optional[Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):  # BxCxHxW
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # BxHxWxC
        x, _ = self.norm1(x, emb, class_labels)
        x = x.permute(0, 3, 1, 2)  # BxCxHxW
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)  # BxHxWxC
        x, _ = self.norm2(x, emb, class_labels)
        x = x.permute(0, 3, 1, 2)  # BxCxHxW
        return x


class GatedCNNBlock(nn.Module):
    """Our implementation from https://github.com/yuweihao/MambaOut"""
    def __init__(
        self,
        dim,
        embed_dim: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        expansion_ratio=8/3,
        kernel_size=7,
        conv_ratio=1.0,
        drop_path=0.,
    ):  # 
        super().__init__()
        self.norm = AdaLayerNorm(dim, embed_dim, num_embeddings, adaLN_Zero=False)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self,
        x: Tensor,
        emb: Optional[Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        shortcut = x # BxHxWxC
        x, _ = self.norm(x, emb, class_labels)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # BxHxWxC -> BxCxHxW
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # BxCxHxW -> BxHxWxC
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut # BxHxWxC


class EfficientUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):  # BxCxHxW
        x = self.conv(x)                # (B, C*r^2, H, W)
        x = self.pixel_shuffle(x)       # (B, C, H*r, W*r)
        return x
    
    
class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        block_out_channels: tuple[int, ...] = (64, 128, 256),
        blocks_per_scale: tuple[int, ...] = (2, 2, 3),
        num_class_embeds: Optional[int] = None,  # Enable conditioning
        num_refine_blocks: int = 2,
        local_embeds: bool = False,  # Each Norm Layer learns its own embedding table
        drop_path: float = 0.,
        with_stem: bool = False,
        add_residual: bool = True,
    ):
        super().__init__()
        if len(block_out_channels) != len(blocks_per_scale):
            raise ValueError("Must provide same number of `block_out_channels` and `blocks_per_scale`")

        self.add_residual = add_residual
        
        self.local_embeds = local_embeds
        embed_dim = None
        self.embedding = None
        num_embeds = None
        if num_class_embeds is not None:
            embed_dim = block_out_channels[0] * 4
            if local_embeds is False:
                self.embedding = nn.Embedding(num_class_embeds, embed_dim)
            else:
                num_embeds = num_class_embeds
                
        if with_stem:
            self.conv_in = StemLayer(
                in_channels,
                block_out_channels[0],
                embed_dim,
                num_embeds,
            )
            self.conv_out = EfficientUpsampleBlock(
                block_out_channels[0],
                in_channels,
                upscale_factor=4,
            )
        else:
            self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)  
            self.conv_out = nn.Conv2d(block_out_channels[0], in_channels, 3, 1, 1)

        self.down_blocks = nn.ModuleList([])
        self.downsample_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        self.skip_projections = nn.ModuleList([])
        self.refine_block = None
        
        # down, stride of 2^(len(block_out_channels) - 1)
        for i in range(len(block_out_channels) - 1):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i + 1]

            blocks = []
            for _ in range(blocks_per_scale[i]):
                blocks.append(GatedCNNBlock(
                    in_channels,
                    embed_dim,
                    num_embeds,
                    drop_path=drop_path,
                ))
            self.down_blocks.append(CondSequential(*blocks))

            self.downsample_blocks.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))

        # mid, use last block_out_channels and blocks_per_scale
        blocks = []
        for _ in range(blocks_per_scale[-1]):
            blocks.append(GatedCNNBlock(
                out_channels,
                embed_dim,
                num_embeds,
                drop_path=drop_path,
            ))
        self.mid_block = CondSequential(*blocks)

        # up, till input resolution of down again
        for i in reversed(range(1, len(block_out_channels))):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i - 1]

            self.upsample_blocks.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))

            self.skip_projections.append(nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))

            blocks = []
            for _ in range(blocks_per_scale[i - 1]):
                blocks.append(GatedCNNBlock(
                    out_channels,
                    embed_dim,
                    num_embeds,
                    drop_path=drop_path,
                ))
            self.up_blocks.append(CondSequential(*blocks))

        if num_refine_blocks > 0:
            blocks = []
            for _ in range(num_refine_blocks):
                blocks.append(GatedCNNBlock(
                    out_channels,
                    embed_dim,
                    num_embeds,
                    drop_path=drop_path,
                ))
            self.refine_block = CondSequential(*blocks)
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"UNet with {1e-6 * n_params:.3f} M parameters.")
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def forward(
        self,
        x: Tensor,
        emb: Optional[Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        x_input = x  # BxCxHxW

        x = self.conv_in(x)  # BxCxHxW

        if emb is not None:
            emb = self.embedding(emb)

        residuals = ()
        for i in range(len(self.down_blocks)):
            x = x.permute(0, 2, 3, 1)  # BxHxWxC
            x = self.down_blocks[i](x, emb, class_labels)  
            x = x.permute(0, 3, 1, 2)  # BxCxHxW
            residuals += (x,)  
            x = self.downsample_blocks[i](x)

        x = x.permute(0, 2, 3, 1)  # BxHxWxC
        x = self.mid_block(x, emb, class_labels)
        x = x.permute(0, 3, 1, 2)  # BxCxHxW

        residuals = tuple(reversed(residuals))
        for i in range(len(self.up_blocks)):
            x = self.upsample_blocks[i](x)
            x = torch.concat((x, residuals[i]), 1)
            x = self.skip_projections[i](x)
            x = x.permute(0, 2, 3, 1)  # BxHxWxC
            x = self.up_blocks[i](x, emb, class_labels)
            x = x.permute(0, 3, 1, 2)  # BxCxHxW

        if self.refine_block is not None:
            x = x.permute(0, 2, 3, 1)  # BxHxWxC
            x = self.refine_block(x, emb, class_labels)
            x = x.permute(0, 3, 1, 2)  # BxCxHxW
        
        x = self.conv_out(x)

        if self.add_residual:
            x = x + x_input

        return x  # BxCxHxW
        
        
if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128).cuda()
    emb = torch.randint(0, 32, (2,)).cuda()
    class_labels = torch.randint(0, 32, (2,)).cuda()
    
    model = UNet(1, with_stem=True).cuda()
    y = model(x, None)
    print(y.shape)
    
    model = UNet(1, num_class_embeds=32, with_stem=True).cuda()
    y = model(x, emb)
    print(y.shape)

    model = UNet(1, (32, 64, 96), (2, 2, 3), None).cuda()
    y = model(x, None)
    print(y.shape)

    model = UNet(1, (32, 64, 96), (2, 2, 3), num_class_embeds=32).cuda()
    y = model(x, emb)
    print(y.shape)
    
    model = UNet(1, (32, 64, 96), (2, 2, 3), num_class_embeds=32, local_embeds=True).cuda()
    y = model(x, None, class_labels)
    print(y.shape)
    