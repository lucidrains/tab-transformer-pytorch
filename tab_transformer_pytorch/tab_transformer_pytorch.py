import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_categories,
        dim,
        depth
    ):
        super().__init__()

    def forward(self, x_categ, x_cont):
        return 0.
