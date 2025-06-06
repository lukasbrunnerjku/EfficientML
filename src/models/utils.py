import torch
import torch.nn as nn
from typing import Optional


class CondSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *cond):
        for layer in self.layers:
            if len(cond) > 0:
                x = layer(x, *cond)
            else:
                x = layer(x)
        return x
    
    
class AdaLayerNorm(nn.Module):
    r"""
    This custom implementation expects input to be BxHxWxC to work.
    Norm layer adaptive layer norm zero (adaLN-Zero) from DiT.

    Parameters:
        in_channels (`int`): The num. channels of the input image x.
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        adaLN_Zero (`bool`): If False adaLN, if True adaLN-Zero.

    The adaLN-Zero computes an additional scale parameter. The DiT block
    first applies layer normalization, scales and shifts this output and then
    feeds it to MHSA/FFN and at the end scales the output of MHSA/FFN with the
    other scale it computed from the condition. Thus, if adaLN_Zero=True the linear layer
    predicts a vector with 3x in_channels dimension (otherwise 2x in_channels). 

    If num_embeddings is given we learn an embedding table for each normalization layer.
    If embedding_dim is not given, we only apply a regular LayerNorm (non-adaptive).
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        adaLN_Zero: bool = False,
        bias: bool = True,  # The linear layer bias for embedding transformation
    ):
        super().__init__()
        self.emb = None
        if embedding_dim is None:  # Not adaptive
            self.norm = nn.LayerNorm(in_channels, elementwise_affine=True, eps=1e-6)
        else:  # Adaptive
            if num_embeddings is not None:
                self.emb = nn.Embedding(num_embeddings, embedding_dim)

            self.silu = nn.SiLU()
            self.multiplier = 3 if adaLN_Zero else 2
            self.linear = nn.Linear(embedding_dim, self.multiplier * in_channels, bias=bias)
            self.norm = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expects x of shape BxHxWxC and also returns x in this shape.
        Gate is either scalar 1.0 or of shape BxCx1x1.
        """
        if self.emb is not None:  # AdaLayerNorm with individual embeddings.
            emb = self.emb(class_labels)

        if emb is not None:  # AdaLayerNorm with shared embeddings.
            emb = self.linear(self.silu(emb))
            if self.multiplier == 2:
                shift, scale = emb.chunk(self.multiplier, dim=1)  # BxC
                gate = 1.0
            else:
                shift, scale, gate = emb.chunk(self.multiplier, dim=1)  # BxC
                gate = gate[:, :, None, None]  # BxCx1x1
            x = self.norm(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
            return x, gate
        else:  # Just a regular LayerNorm.
            x = self.norm(x)
            return x, 1.0
        
