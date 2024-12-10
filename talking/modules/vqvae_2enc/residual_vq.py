import torch
from torch import nn
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import lightning.pytorch as pl

class ResidualVQ(pl.LightningModule):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantizer(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_out = 0.
        residual = x
        
        all_losses = []
        # all_indices = []

        for idx, layer in enumerate(self.layers):
            quantized, loss, _ = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            # print(idx, torch.abs(x - quantized_out).mean())

            # all_indices.append(indices)
            all_losses.append(loss)

        # all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        all_losses = torch.stack(all_losses)
        return quantized_out, all_losses, _

    def vq2emb(self, vq):
        quantized_out = 0.
        for idx, layer in enumerate(self.layers):
            quantized = layer.vq2emb(vq[:,idx])
            quantized_out = quantized_out + quantized
        return quantized_out
    
    def get_emb(self):
        embs = [] 
        for idx, layer in enumerate(self.layers):
            embs.append(layer.get_emb())
        return embs
