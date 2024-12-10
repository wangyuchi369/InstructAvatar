# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing import Optional, Any

from ldm.modules.attention import MemoryEfficientCrossAttention

try:
    import xformers
    import xformers.ops
    device_name = torch.cuda.get_device_name(0)
    # xformer should be compiled from source
    # Pascal (sm60), Volta (sm70), Turing (sm75), Ampere (sm80)
    XFORMERS_IS_AVAILBLE = True
    print(f"[temporal_modules] Using XFORMERS with {device_name}.")
except:
    XFORMERS_IS_AVAILBLE = False
    print("[temporal_modules] No module 'xformers'. Proceeding without it.")



def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class AttnBlock_Spatio_Temporal(AttnBlock):
    def __init__(self, in_channels, temporal_len=5):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        # TODO as a parameter
        self.temporal_len=temporal_len
        self.norm_temp = Normalize(in_channels)
        self.q_temp = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k_temp = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v_temp = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out_temp = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
       

        spatio_out = super().forward(x)

        # now begin temporal attention
        h_ = spatio_out
        h_ = self.norm_temp(h_)
        q = self.q_temp(h_)
        k = self.k_temp(h_)
        v = self.v_temp(h_)  # same shape

        # compute attention
        bt, c, h, w = q.shape
        t= self.temporal_len
        b=bt//t
        # q = q.reshape(b, c, h * w)
        # q = q.permute(0, 2, 1)  # b,hw,c
        q = rearrange(q, '(b t) c h w -> (b h w) t c',t=self.temporal_len)
        # k = k.reshape(b, c, h * w)  # b,c,hw
        k = rearrange(k, '(b t) c h w -> (b h w) c t',t=self.temporal_len)
        w_ = torch.bmm(q, k)  #    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        #v = v.reshape(b *h * w, c, t)
        v = rearrange(v, '(b t) c h w  -> (b h w) c t', t=t)
        w_ = w_.permute(0, 2, 1)  # bhw,t,t 
        h_ = torch.bmm(v, w_)  # bhw, c, t
        assert bt == b*t
        #h_ = h_.reshape(b*t, c, h, w)
        h_ = rearrange(h_, '(b h w) c t -> (b t) c h w', t=t,h=h,w=w)

        h_ = self.proj_out_temp(h_)
        return x+h_

class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
  
        out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x+out

class MemoryEfficientAttnBlockWithTemporal(MemoryEfficientAttnBlock):
    
    def __init__(self, in_channels, temporal_len=5):
        super().__init__(in_channels=in_channels)
        
        # TODO as a parameter
        self.temporal_len=temporal_len
        self.norm_temp = Normalize(in_channels)
        self.q_temp = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k_temp = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v_temp = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out_temp = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        spatio_out = super().forward(x)
         # now begin temporal attention
        h_ = spatio_out
        h_ = self.norm_temp(h_)
        q = self.q_temp(h_)
        k = self.k_temp(h_)
        v = self.v_temp(h_)  # same shape

        # compute attention
        bt, c, h, w = q.shape
        t= self.temporal_len
        b=bt//t
        # q = q.reshape(b, c, h * w)
        # q = q.permute(0, 2, 1)  # b,hw,c
        q, k, v = map(lambda x: rearrange(x, '(b t) c h w -> (b h w) t c',t=t).contiguous(), (q, k, v))
        
        out_temp = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        h_ = rearrange(out_temp, '(b h w) t c -> (b t) c h w', t=t,h=h,w=w)

        h_ = self.proj_out_temp(h_)
        return x+h_

class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out


def make_attn_temporal(in_channels, attn_type="vanilla", temporal_len=None, use_temporal=False,attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"[make_attn] making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla" and not use_temporal:
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla" and use_temporal:
        assert attn_kwargs is None
        return AttnBlock_Spatio_Temporal(in_channels, temporal_len)
    elif attn_type == "vanilla-xformers" and not use_temporal:
        print(f"[make_attn] building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "vanilla-xformers" and use_temporal:
        print(f"[make_attn] building MemoryEfficientAttnBlockWithTemporal with {in_channels} in_channels...")
        return MemoryEfficientAttnBlockWithTemporal(in_channels, temporal_len)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()