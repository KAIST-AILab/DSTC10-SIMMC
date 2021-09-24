import copy
from typing import List, Dict ,Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel
import ipdb
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderLayer(nn.Module):
    def __init__(
        self, d_model=256, 
        n_head=4, 
        dim_feedforward=2048, 
        dropout=0.1, 
        activation=F.relu,
        normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2    

class Encoder(nn.Module):
    def __init__(
        self, encoder_layer, num_layers=3, norm=None # or LayerNorm
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        return output

class DecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward, dropout=0.1, activation=F.relu, normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self, tgt, memory, 
        tgt_mask: Optional[Tensor] = None, 
        memory_mask : Optional[Tensor] = None,
        tgt_key_padding_mask : Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        q = k = self.with_pos_embed(tgt, query_pos)

        # Self Attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
    
        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm=None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self, tgt, memory, 
        tgt_mask : Optional[Tensor] = None,
        memory_mask : Optional[Tensor] = None,
        tgt_key_padding_mask : Optional[Tensor] = None,
        memory_key_padding_mask : Optional[Tensor] = None,
        pos : Optional[Tensor] = None, # memory position encoding
        query_pos : Optional[Tensor] = None,
        return_intermediate = False
    ):
        output = tgt
        intermediate = []
        for layer in self.layers :
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if return_intermediate:
                intermediate.append(output)

        if return_intermediate:
            return torch.stack(intermediate)
        
        return output

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class MultiModalTransformer(nn.Module):

    def __init__(
        self,
        backbone,
        img_feature_dim=2048,
        d_model=512,
        n_head=8,
        num_encoder_layers=6, 
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        normalize_before=False,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        ):
        super().__init__()
        # image backbone
        self.image_encoder = backbone
        self.embedding_to_encoder_inp = nn.Conv2d(img_feature_dim, d_model, kernel_size=1)

        # Endocer
        encoder_layer = EncoderLayer(d_model, n_head, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder 
        decoder_layer = DecoderLayer(d_model, n_head, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)

        # param    
        self.d_model = d_model
        self.n_head = n_head

        # Init
        self._reset_parameters()

        # text encoder
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
    
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.required_grad(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
                input_feat_size=config.hidden_size,
                output_feat_size=d_model,
                dropout=self.expander_dropout,
            )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        images, # NestedTensor .tensors, .mask 
        text : Dict, 
        memory=None,
        query_embedd=None,
        tgt_mask=None,
        memory_mask=None,

        ):

        fea, img_pos = self.image_encoder(images)
        img_src, img_mask = fea[-1].decompose()
        
        # bs, d_model, h, w 
        # -> bs, d_model, h*w 
        # -> h*w, bs, d_model 
        img_src = self.embedding_to_encoder_inp(img_src).flatten(2).permute(2, 0, 1)
        img_pos = img_pos[-1].flatten(2).permute(2, 0, 1)
        img_mask = img_mask.flatten(1) # bs, h*w
        img_src = img_src + 0.1 * img_pos

        try:
            text_src = self.text_encoder(**text).last_hidden_state.permute(1, 0, 2) # seq_len, bs, dim
        except:
            ipdb.set_trace()
        text_src = self.resizer(text_src) # seq_len, bs, d_model
        text_mask = text.attention_mask # bs, seq_len

        mask = torch.cat([img_mask, text_mask], dim=-1) # bs, text_seq_len + img_seq_len
        src = torch.cat([img_src, text_src], dim=0) # text_seq_len + img_seq_len, bs, d_model
        memory = self.encoder(src, src_key_padding_mask=mask, pos=None) 

        # decode
        tgt = self.resizer(query_embedd).permute(1, 0, 2)
        pos_embed = None
        query_pos = None
        # tgt_mask and memory_mask check!! --> feedforward to attn_mask  
        # tgt_mask
        # o 1) tgt padding with "<@0>" None object without query_mask
        # x 2) tgt padding with "<@0>" None object with query_mask
        # memory mask
        # 1) using only memory_key_padding_mask, memory_mask = None 

        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            memory_key_padding_mask = mask,
            pos=pos_embed,
            query_pos=query_pos,
        )

        return output 
    
    def encode(self, images, text: Dict):
        fea, img_pos = self.image_encoder(images)
        img_src, img_mask = fea[-1].decompose()
        
        # bs, d_model, h, w 
        # -> bs, d_model, h*w 
        # -> h*w, bs, d_model 
        img_src = self.embedding_to_encoder_inp(img_src).flatten(2).permute(2, 0, 1)
        img_pos = img_pos[-1].flatten(2).permute(2, 0, 1)
        img_mask = img_mask.flatten(1) # bs, h*w
        img_src = img_src + 0.1 * img_pos

        text_src = self.text_encoder(**text).last_hidden_state.permute(1, 0, 2) # seq_len, bs, dim
        text_src = self.resizer(text_src) # seq_len, bs, d_model
        text_mask = text.attention_mask # bs, seq_len

        mask = torch.cat([img_mask, text_mask], dim=-1) # bs, text_seq_len + img_seq_len
        src = torch.cat([img_src, text_src], dim=0) # text_seq_len + img_seq_len, bs, d_model
        memory = self.encoder(src, src_key_padding_mask=mask, pos=None)
        return memory, mask
    
    def decode(self, query_embedd, memory, tgt_mask=None, memory_mask=None, memory_key_padding_mask=None):
        tgt = self.resizer(query_embedd).permute(1, 0, 2)
        pos_embed = None
        query_pos = None        
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            memory_key_padding_mask = memory_key_padding_mask,
            pos=pos_embed,
            query_pos=query_pos,
        )
        return output 

class UniModalTransformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        n_head=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        normalize_before=False,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        ):
        super().__init__()

        # Decoder 
        decoder_layer = DecoderLayer(d_model, n_head, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)

        # param    
        self.d_model = d_model
        self.n_head = n_head

        # Init
        self._reset_parameters()

        # text encoder
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
    
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.required_grad(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
                input_feat_size=config.hidden_size,
                output_feat_size=d_model,
                dropout=self.expander_dropout,
            )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        text : Dict, 
        memory=None,
        query_embedd=None,
        tgt_mask=None,
        memory_mask=None,

        ):

        memory = self.text_encoder(**text).last_hidden_state.permute(1, 0, 2) # seq_len, bs, dim
        memory = self.resizer(memory) # seq_len, bs, d_model
        mask = text.attention_mask # bs, seq_len

        # decode
        tgt = self.resizer(query_embedd).permute(1, 0, 2)
        pos_embed = None
        query_pos = None
        # tgt_mask and memory_mask check!! --> feedforward to attn_mask  
        # tgt_mask
        # o 1) tgt padding with "<@0>" None object without query_mask
        # x 2) tgt padding with "<@0>" None object with query_mask
        # memory mask
        # 1) using only memory_key_padding_mask, memory_mask = None 

        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            memory_key_padding_mask = mask,
            pos=pos_embed,
            query_pos=query_pos,
        )

        return output 
    
    def encode(self, images, text: Dict):
        memory = self.text_encoder(**text).last_hidden_state.permute(1, 0, 2) # seq_len, bs, dim
        memory = self.resizer(src) # seq_len, bs, d_model
        mask = text.attention_mask # bs, seq_len
        return memory, mask
    
    def decode(self, query_embedd, memory, tgt_mask=None, memory_mask=None, memory_key_padding_mask=None):
        tgt = self.resizer(query_embedd).permute(1, 0, 2)
        pos_embed = None
        query_pos = None        
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            memory_key_padding_mask = memory_key_padding_mask,
            pos=pos_embed,
            query_pos=query_pos,
        )
        return output 