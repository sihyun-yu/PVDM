import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.autoencoder.vit_modules import TimeSformerEncoder, TimeSformerDecoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from time import sleep

# siren layer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ===========================================================================================


class ViTAutoencoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.splits = ddconfig["splits"]
        self.s = ddconfig["timesteps"] // self.splits 

        self.res = ddconfig["resolution"]
        self.embed_dim = embed_dim
        self.image_key = image_key

        patch_size = 8
        if self.res == 128:
            patch_size = 4
        self.down = 3

        self.encoder = TimeSformerEncoder(dim=ddconfig["channels"],
                                   image_size=ddconfig["resolution"],
                                   num_frames=ddconfig["timesteps"],
                                   depth=8,
                                   patch_size=patch_size)

        self.decoder = TimeSformerDecoder(dim=ddconfig["channels"],
                                   image_size=ddconfig["resolution"],
                                   num_frames=ddconfig["timesteps"],
                                   depth=8,
                                   patch_size=patch_size)

        self.to_pixel = nn.Sequential(
            Rearrange('b (t h w) c -> (b t) c h w', h=self.res // patch_size, w=self.res // patch_size),
            nn.ConvTranspose2d(ddconfig["channels"], 3, kernel_size=(patch_size, patch_size), stride=patch_size),
            )          

        self.act = nn.Sigmoid()
        ts = torch.linspace(-1, 1, steps=self.s).unsqueeze(-1)
        self.register_buffer('coords', ts)

        self.xy_token = nn.Parameter(torch.randn(1, 1, ddconfig["channels"]))
        self.xt_token = nn.Parameter(torch.randn(1, 1, ddconfig["channels"]))
        self.yt_token = nn.Parameter(torch.randn(1, 1, ddconfig["channels"]))

        self.xy_pos_embedding = nn.Parameter(torch.randn(1, self.s + 1, ddconfig["channels"]))
        self.xt_pos_embedding = nn.Parameter(torch.randn(1, self.res//(2**self.down) + 1, ddconfig["channels"]))
        self.yt_pos_embedding = nn.Parameter(torch.randn(1, self.res//(2**self.down) + 1, ddconfig["channels"]))

        self.xy_quant_attn = Transformer(ddconfig["channels"], 4, self.embed_dim, ddconfig["channels"] // 8, 512)
        self.yt_quant_attn = Transformer(ddconfig["channels"], 4, self.embed_dim, ddconfig["channels"] // 8, 512)
        self.xt_quant_attn = Transformer(ddconfig["channels"], 4, self.embed_dim, ddconfig["channels"] // 8, 512)

        self.pre_xy = torch.nn.Conv2d(ddconfig["channels"], self.embed_dim, 1)
        self.pre_xt = torch.nn.Conv2d(ddconfig["channels"], self.embed_dim, 1)
        self.pre_yt = torch.nn.Conv2d(ddconfig["channels"], self.embed_dim, 1)

        self.post_xy = torch.nn.Conv2d(self.embed_dim, ddconfig["channels"], 1)
        self.post_xt = torch.nn.Conv2d(self.embed_dim, ddconfig["channels"], 1)
        self.post_yt = torch.nn.Conv2d(self.embed_dim, ddconfig["channels"], 1)

    def encode(self, x):
        # x: b c t h w
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> b t c h w')
        h = self.encoder(x)
        h = rearrange(h, 'b (t h w) c -> b c t h w', t=self.s, h=self.res//(2**self.down))

        h_xy = rearrange(h, 'b c t h w -> (b h w) t c')
        n = h_xy.size(1)
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw = h_xy.size(0))
        h_xy = torch.cat([h_xy, xy_token], dim=1)
        h_xy += self.xy_pos_embedding[:, :(n+1)]
        h_xy = self.xy_quant_attn(h_xy)[:, 0]
        h_xy = rearrange(h_xy, '(b h w) c -> b c h w', b=b, h=self.res//(2**self.down))

        h_yt = rearrange(h, 'b c t h w -> (b t w) h c')
        n = h_yt.size(1)
        yt_token = repeat(self.yt_token, '1 1 d -> btw 1 d', btw = h_yt.size(0))
        h_yt = torch.cat([h_yt, yt_token], dim=1)
        h_yt += self.yt_pos_embedding[:, :(n+1)]
        h_yt = self.yt_quant_attn(h_yt)[:, 0]
        h_yt = rearrange(h_yt, '(b t w) c -> b c t w', b=b, w=self.res//(2**self.down))

        h_xt = rearrange(h, 'b c t h w -> (b t h) w c')
        n = h_xt.size(1)
        xt_token = repeat(self.xt_token, '1 1 d -> bth 1 d', bth = h_xt.size(0))
        h_xt = torch.cat([h_xt, xt_token], dim=1)
        h_xt += self.xt_pos_embedding[:, :(n+1)]
        h_xt = self.xt_quant_attn(h_xt)[:, 0]
        h_xt = rearrange(h_xt, '(b t h) c -> b c t h', b=b, h=self.res//(2**self.down))

        h_xy = self.pre_xy(h_xy)
        h_yt = self.pre_yt(h_yt)
        h_xt = self.pre_xt(h_xt)

        h_xy = torch.tanh(h_xy)
        h_yt = torch.tanh(h_yt)
        h_xt = torch.tanh(h_xt)

        h_xy = self.post_xy(h_xy)
        h_yt = self.post_yt(h_yt)
        h_xt = self.post_xt(h_xt)

        h_xy = h_xy.unsqueeze(-3).expand(-1,-1,self.s,-1, -1)
        h_yt = h_yt.unsqueeze(-2).expand(-1,-1,-1,self.res//(2**self.down), -1)
        h_xt = h_xt.unsqueeze(-1).expand(-1,-1,-1,-1,self.res//(2**self.down))

        return h_xy + h_yt + h_xt #torch.cat([h_xy, h_yt, h_xt], dim=1)

    def decode(self, z):
        b = z.size(0)
        dec = self.decoder(z)
        return 2*self.act(self.to_pixel(dec)).contiguous() -1

    def forward(self, input):
        input = rearrange(input, 'b c (n t) h w -> (b n) c t h w', n=self.splits)
        z = self.encode(input)
        dec = self.decode(z)
        return dec, 0.

    def extract(self, x):
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> b t c h w')
        h = self.encoder(x)
        h = rearrange(h, 'b (t h w) c -> b c t h w', t=self.s, h=self.res//(2**self.down))

        h_xy = rearrange(h, 'b c t h w -> (b h w) t c')
        n = h_xy.size(1)
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw = h_xy.size(0))
        h_xy = torch.cat([h_xy, xy_token], dim=1)
        h_xy += self.xy_pos_embedding[:, :(n+1)]
        h_xy = self.xy_quant_attn(h_xy)[:, 0]
        h_xy = rearrange(h_xy, '(b h w) c -> b c h w', b=b, h=self.res//(2**self.down))

        h_yt = rearrange(h, 'b c t h w -> (b t w) h c')
        n = h_yt.size(1)
        yt_token = repeat(self.yt_token, '1 1 d -> btw 1 d', btw = h_yt.size(0))
        h_yt = torch.cat([h_yt, yt_token], dim=1)
        h_yt += self.yt_pos_embedding[:, :(n+1)]
        h_yt = self.yt_quant_attn(h_yt)[:, 0]
        h_yt = rearrange(h_yt, '(b t w) c -> b c t w', b=b, w=self.res//(2**self.down))

        h_xt = rearrange(h, 'b c t h w -> (b t h) w c')
        n = h_xt.size(1)
        xt_token = repeat(self.xt_token, '1 1 d -> bth 1 d', bth = h_xt.size(0))
        h_xt = torch.cat([h_xt, xt_token], dim=1)
        h_xt += self.xt_pos_embedding[:, :(n+1)]
        h_xt = self.xt_quant_attn(h_xt)[:, 0]
        h_xt = rearrange(h_xt, '(b t h) c -> b c t h', b=b, h=self.res//(2**self.down))

        h_xy = self.pre_xy(h_xy)
        h_yt = self.pre_yt(h_yt)
        h_xt = self.pre_xt(h_xt)

        h_xy = torch.tanh(h_xy)
        h_yt = torch.tanh(h_yt)
        h_xt = torch.tanh(h_xt)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        ret =  torch.cat([h_xy, h_yt, h_xt], dim=-1)
        return ret 

    def decode_from_sample(self, h):
        latent_res = self.res // (2**self.down)
        h_xy = h[:, :, 0:latent_res*latent_res].view(h.size(0), h.size(1), latent_res, latent_res)
        h_yt = h[:, :, latent_res*latent_res:latent_res*(latent_res+16)].view(h.size(0), h.size(1), 16, latent_res)
        h_xt = h[:, :, latent_res*(latent_res+16):latent_res*(latent_res+16+16)].view(h.size(0), h.size(1), 16, latent_res)

        h_xy = self.post_xy(h_xy)
        h_yt = self.post_yt(h_yt)
        h_xt = self.post_xt(h_xt)

        h_xy = h_xy.unsqueeze(-3).expand(-1,-1,self.s,-1, -1)
        h_yt = h_yt.unsqueeze(-2).expand(-1,-1,-1,self.res//(2**self.down), -1)
        h_xt = h_xt.unsqueeze(-1).expand(-1,-1,-1,-1,self.res//(2**self.down))

        z = h_xy + h_yt + h_xt

        b = z.size(0)
        dec = self.decoder(z)
        return 2*self.act(self.to_pixel(dec)).contiguous()-1
