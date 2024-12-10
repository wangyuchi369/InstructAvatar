import torch
import torch.nn as nn
"""
Applies the mish function element-wise:
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
"""

# import pytorch
import torch
import torch.nn.functional as F
from torch import nn

@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if torch.__version__ >= "1.9":
            return F.mish(input)
        else:
            return mish(input)


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Sequential(nn.Linear(input_dim, input_dim), Mish(), nn.Linear(input_dim, 1))
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            batch_rep : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        att_mask:
            att_mask: size (N, T): if True, mask this item.
        return:
            utter_rep: size (N, H)
        """

        att_logits = self.W(batch_rep).squeeze(-1)
        # (N, T)
        if att_mask is not None:
            att_mask_logits = att_mask.to(dtype=batch_rep.dtype) * -100000.0
            # (N, T)
            att_logits = att_mask_logits + att_logits

        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

