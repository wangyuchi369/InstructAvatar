

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
    
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModelWithProjection.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt").to(self.device)
        # tokens = batch_encoding["input_ids"].to(self.device)
        # outputs = self.transformer(input_ids=tokens)
        outputs = self.transformer(**batch_encoding)
        z = outputs.last_hidden_state
        text_embed = outputs.text_embeds
        return z, text_embed

    # extract all hidden states of CLIP
    def encode(self, text):
        return self(text)[0]
    
    # extract text embedding of CLIP
    def encode_text(self, text):
        return self(text)[1]
    
# import clip    
# class FrozenCLIPTextEmbedder(nn.Module):
#     """
#     Uses the CLIP transformer encoder for text.
#     """
#     def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
#         super().__init__()
#         self.model, _ = clip.load(version, jit=False, device="cpu")
#         self.device = device
#         self.max_length = max_length
#         self.n_repeat = n_repeat
#         self.normalize = normalize

#     def freeze(self):
#         self.model = self.model.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         tokens = clip.tokenize(text).to(self.device)
#         z = self.model.encode_text(tokens)
#         if self.normalize:
#             z = z / torch.linalg.norm(z, dim=1, keepdim=True)
#         return z

#     def encode(self, text):
#         z = self(text)
#         if z.ndim==2:
#             z = z[:, None, :]
#         z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
#         return z