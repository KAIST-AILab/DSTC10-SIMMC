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
from dataloader import EvalDataset, mask_tokens
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

import ipdb
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
    eval_dataset = EvalDataset(tokenizer, "/home/yschoi/SIMMC2/retrieval/postprocessing_data.json", )
    sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=sampler, batch_size=1, num_workers=1)
    
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

    model.load_state_dict(torch.load("/home/yschoi/SIMMC2/retrieval/checkpoints/0-2000.pkl"))
    model.to(device)
    model.freeze_text_decoder()
    model.eval()
    mean_top1 = []
    mean_top3 = []
    mean_top5 = []

    gt_labels = []

    for batch in tqdm(eval_dataloader, "DevTesting..."):
        ipdb.set_trace()
        scores = []
        for idx in range(0, 100, 2):
            input_ids, labels = mask_tokens(batch["candidates"]["input_ids"][0][idx: idx+2], tokenizer, 0.15)
            input_ids, labels, attention_masks \
                = move_to_device([input_ids, labels, batch["candidates"]["attention_mask"][0][idx:idx+2]], device)

            loss_score = model.forward_mlm(
                    input_ids=input_ids,
                    attention_mask = attention_masks,
                    labels=labels,
                    reduce=False,
                ) # 
            scores.append(loss_score.view(2, -1).mean(-1).cpu())
        
        loss_scores =torch.cat(scores, dim=0)
        loss_scores = loss_scores.view(100, -1).cpu()
        # scores_sum = torch.sum(loss_scores, dim=-1).cpu()
        scores_mean = torch.mean(loss_scores, dim=-1).cpu()

        values, indices = torch.topk(scores_mean, 1, largest=False)
        mean_top1.append(indices)
        values, indices = torch.topk(scores_mean, 3, largest=False)
        mean_top3.append(indices)
        values, indices = torch.topk(scores_mean, 5, largest=False)
        mean_top5.append(indices)

        gt_labels.append(batch["gt_index"])


    indices =torch.cat(mean_top5, dim=0).long() # bs, num_indic
    label = torch.tensor(gt_lables).reshpae(-1, 1).long() # bs
    top5_value = (indices == label).any(1).float().mean()
    pritn(top5_value)