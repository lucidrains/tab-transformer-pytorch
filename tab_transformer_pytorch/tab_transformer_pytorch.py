import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_unique_categories,
        num_categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2)
    ):
        super().__init__()
        assert len(mlp_hidden_mults) == 2, 'final mlp fixed at 2 layers for now'

        self.categorical_embeds = nn.Embedding(num_unique_categories, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head))),
                Residual(PreNorm(dim, FeedForward(dim))),
            ]))

        self.norm = nn.LayerNorm(num_continuous)

        input_size = (dim * num_categories) + num_continuous
        l = input_size // 8
        mult1, mult2 = mlp_hidden_mults

        self.mlp = nn.Sequential(
            nn.Linear(input_size, l * mult1),
            nn.ReLU(),
            nn.Linear(l * mult1, l * mult2),
            nn.ReLU(),
            nn.Linear(l * mult2, dim_out)
        )

    def forward(self, x_categ, x_cont):
        x = self.categorical_embeds(x_categ)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        flat_categ = x.flatten(1)
        normed_cont = self.norm(x_cont)

        x = torch.cat((flat_categ, normed_cont), dim = -1)
        return self.mlp(x)
