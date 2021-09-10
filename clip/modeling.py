import torch
import numpy as np

from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F

from transformers.file_utils import ModelOutput


@dataclass
class CLIPOutput(ModelOutput):
    '''
        Data class for CLIP output. Inherits from ModelOutput, which can be iterated over itself.
    '''
    metadata_embedding: torch.FloatTensor = None
    object_embedding: torch.FloatTensor = None
    metadata_loss: torch.FloatTensor = None
    object_loss: torch.FloatTensor = None
    total_loss: torch.FloatTensor = None


class AttentionPooler(nn.Module):
    def __init__(self, head_size: int, attention_dim: int, pool_type='mean'):
        assert pool_type in ('mean', 'cls')
        super().__init__()

        self.pool_type = pool_type
        self.std = attention_dim ** 0.5
        self.cls_token = nn.Parameter(torch.randn([1, 1, head_size])).float()
        self.head_size = head_size
        self.attention_dim = attention_dim

        self.WQ = nn.Linear(head_size, attention_dim, bias=False)
        self.WK = nn.Linear(head_size, attention_dim, bias=False)
        self.WV = nn.Linear(head_size, attention_dim, bias=False)

    def forward(self, x, mask=None):
        if self.pool_type == 'cls':
            cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tok, x], dim=1)
            cls_msk = torch.zeros([x.shape[0], 1], device=x.device)
            # [batch_sz, seq_len + 1, 1]
            mask = torch.cat([cls_msk, mask], dim=1).bool()
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        a = torch.einsum('b i d, b j d -> b i j', q, k) / self.std        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            a = a.masked_fill(mask, -1e9)
        a = F.softmax(a, dim=-1)
        a = torch.einsum('b i j, b j d -> b i d', a, v)
        if self.pool_type == 'mean':
            if mask is not None:
                a = (a * ~mask).sum(dim=1) / (~mask).sum(dim=1)
        else:
            a = a[:,0,:]
        return a


class MetadataEncoder(nn.Module):
    '''
        Encodes metadata string using GLoVE embedding + attention pooling
    '''
    def __init__(self, attention_dim: int, pool_type="mean", glove_path: str=None):
        super().__init__()

        glove = torch.load(glove_path)

        self.embedding = nn.Embedding.from_pretrained(
            glove, padding_idx=0
        )
        # self.attn_pool = nn.MultiheadAttention(
        #     embed_dim=self.embedding.embedding_dim,
        #     num_heads=4
        # )
        head_size = self.embedding.embedding_dim
        self.attn_pool = AttentionPooler(
            head_size, attention_dim, pool_type
        )
    
    def forward(self, x):
        mask = (x == 0)
        x = self.embedding(x).float()
        x = self.attn_pool(x, mask)
        # x = x.permute(1,0,2)
        # x = self.attn_pool(x, x, x, key_padding_mask=mask, need_weights=False)
        return x


class SimpleCLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Metadata encoder / pooler
        self.metadata_encoder = MetadataEncoder(
            args.attention_dim, args.pool_type, args.glove_path
        )
        self.metadata_pooler = nn.Linear(
            args.attention_dim, args.embedding_dim
        )
        # Object encoder / pooler -- image or item vector
        self.object_encoder = nn.Embedding(
            args.num_object_tokens, args.embedding_dim, padding_idx=0
        )
        self.object_pooler = nn.Linear(
            args.embedding_dim, args.embedding_dim
        )
        # Learnable logit scaler
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 7e-2))

    def encode_metadata(self, dsc):
        dsc = self.metadata_encoder(dsc)
        return self.metadata_pooler(dsc)

    def encode_object(self, obj):
        obj = self.object_encoder(obj)
        return self.object_pooler(obj)

    def forward(self, obj: torch.Tensor, dsc: torch.Tensor):
        # Encode and normalize
        obj = self.encode_object(obj)
        dsc = self.encode_metadata(dsc)
        obj = obj / obj.norm(dim=-1, keepdim=True)
        dsc = dsc / dsc.norm(dim=-1, keepdim=True)
        # Get logits
        logit_scale = self.logit_scale.exp()
        logit_per_obj = logit_scale * obj @ dsc.t()
        logit_per_dsc = logit_scale * dsc @ obj.t()
        # Get loss
        ground_truth = torch.arange(
            len(obj), dtype=torch.long, device=obj.device
        )
        obj_loss = F.cross_entropy(logit_per_obj, ground_truth)
        dsc_loss = F.cross_entropy(logit_per_dsc, ground_truth)
        total_loss = (obj_loss + dsc_loss) * 0.5
        
        return CLIPOutput(
            metadata_embedding=dsc,
            object_embedding=obj,
            metadata_loss=dsc_loss,
            object_loss=obj_loss,
            total_loss=total_loss
        )
