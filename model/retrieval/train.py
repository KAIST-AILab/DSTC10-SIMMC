import os
import time
import uuid
import random
import datetime
import logging
import json
import torch
import numpy as np

from math import ceil
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp
from torch import distributed as dist
from torch.optim import AdamW

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (
    DataLoader, DistributedSampler, RandomSampler, random_split, SequentialSampler
)
from torchvision import transforms

from transformers import (
    BartTokenizerFast, 
    BartModel,
    get_linear_schedule_with_warmup,
    AdamW
)
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from dataloader import Simmc2Dataset, Simmc2Collate, mask_tokens
import pickle
from model import Model
def move_to_device(data, device):
    '''
        Recursive handler to move data to device.

        Args:
            data <Any>: can be torch.Tensor, list, tuple, or dict
            device <torch.device, str>: device to which the data will be placed
        Returns:
            data <Any>: data placed on device
    '''
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        for idx, ele in enumerate(data):
            data[idx] = move_to_device(ele, device)
        return data
    if isinstance(data, tuple):
        return tuple(move_to_device(list(data), device))
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = move_to_device(v, device)
        return data
    return data

if __name__=="__main__":
    device = torch.device("cuda" if not False and torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir=None)
    special_tokens_dict = json.load(open("/home/yschoi/SIMMC2/bart_with_item/special_tokens.json", "r"))
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    vocab_size = len(tokenizer)
    obj_tag_token_id = tokenizer.convert_tokens_to_ids("<OBJ>")
    obj_start_token_id = tokenizer.convert_tokens_to_ids("<SOO>")
    obj_end_token_id = tokenizer.convert_tokens_to_ids("<EOO>")
    train_collate = Simmc2Collate(tokenizer, max_length=256)
    train_dataset = Simmc2Dataset(tokenizer, "/home/yschoi/SIMMC2/data/retrieval/train.json")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=12, collate_fn=train_collate, num_workers=16)

    kwargs = {
        'item_size' : 346,
        'pad_item_idx' : 0,
        'pretrained_item_embedding' : None,
        'loc_layer_input_dim' : 4,
        "obj_tag_token_id" : obj_tag_token_id,
        "obj_start_token_id" : obj_start_token_id,
        "obj_end_token_id" : obj_end_token_id
    }

    print(f"Vocab_size {vocab_size}")
    model = Model.from_pretrained("facebook/bart-large", **kwargs)
    model.resize_token_embeddings(vocab_size)
    model.vocab_size = vocab_size
    model._make_layer_for_pretraining()
    model.config.decoder_start_token_id = 0
    model.to(device)
    
    mlm_pretrained_optimizer = AdamW([
            # {'params' : model.item_embedding.parameters()},
            # {'params' : model.loc_layer.parameters()},
            {'params' : model.model.encoder.parameters()},
            {'params' : model.mlm_cls.parameters()},
    ], lr=5e-5, eps=1e-8)

    t_total = len(train_dataloader)*10
    pretrain_scheduler = get_linear_schedule_with_warmup(
        mlm_pretrained_optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # MLM - Pretraining
    model.freeze_text_decoder()
    step = 0
    model.train()
    for epoch in range(1):
        for batch in tqdm(train_dataloader, f"Epoch {epoch}, MLM Training "):
            input_ids, labels = mask_tokens(batch["encoder_input_ids"], tokenizer, 0.15)
            input_ids, labels, encoder_attention_mask \
                = move_to_device([input_ids, labels, batch["encoder_attention_mask"]], device)
            loss = model.forward_mlm(
                input_ids=input_ids,
                attention_mask = encoder_attention_mask,
                labels=labels
            )
            mlm_pretrained_optimizer.zero_grad()
            loss.backward() # loss
            mlm_pretrained_optimizer.step()
            step += 1
            if (step == 1) or (step % 500 == 0):
                print("*"*20)
                print(loss.item())

            # if step % 2000 == 0:
        torch.save(model.state_dict(), f"./checkpoints/{epoch}-{step}.pkl")






                    
    

