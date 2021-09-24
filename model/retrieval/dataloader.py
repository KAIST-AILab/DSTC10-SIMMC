import ipdb
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Callable, Tuple, Any
from PIL import Image, ImageFile
from functools import lru_cache

class Simmc2Dataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, file_path : str,
    ):
        print(f"Data Directory : {file_path}")
        assert os.path.isfile(file_path)
        with open(file_path, "rb") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
    def __len__(self) -> int : 
        return len(self.data.keys())

    @lru_cache()
    def __getitem__(self, i) -> Tuple[Any, Any] : 
        asset = self.data[str(i)]
        encoder_input = asset["encoder_input"]
        return encoder_input


class Simmc2Collate():

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):

        encoder_input = tuple(map(list, zip(*batch)))[0]
        assert self.tokenizer._pad_token is not None
        encoder_input = self.tokenizer(encoder_input, padding="longest", max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            "encoder_input_ids" : encoder_input.input_ids, 
            "encoder_attention_mask" : encoder_input.attention_mask}


class EvalDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, file_path : str, max_length=256
    ):
        print(f"Data Directory : {file_path}")
        assert os.path.isfile(file_path)
        with open(file_path, "rb") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self) -> int : 
        return len(self.data)

    @lru_cache()
    def __getitem__(self, i) -> Tuple[Any, Any] : 
        asset = self.data[i]
        candidates = self.tokenizer(asset["candidates"], padding="longest", max_length=self.max_length, truncation=True, return_tensors="pt")
        gt_index = -100
        if "gt_lable" in asset:
            gt_index = asset["gt_lable"]

        return {
            "candidates" : candidates, 
            "gt_index" : gt_index
        }




def mask_tokens(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizerBase, mlm_probability : float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices # and True and True
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


if __name__=="__main__":
    # tokenizer = PreTrainedTokenizerBase.from_pretrained("facebook/bart-base")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir=None)
    special_tokens_dict = {
        "eos_token": "<EOS>",
        "additional_special_tokens": [
            "<EOB>",
            "<SOM>",
            "<EOM>",
            "<USR>",
            "<SYS>",
            "<#OBJS>",
            "ID",
            "materials",
            "pattern",
            "type",
            "REQUEST:GET",
            "REQUEST:ADD_TO_CART",
            "availableSizes",
            "REQUEST:COMPARE",
            "INFORM:GET",
            "INFORM:REFINE",
            "INFORM:DISAMBIGUATE",
            "ASK:GET",
            "customerRating",
            "price",
            "brand",
            "color",
            "size",
            "sleeveLength",
            "customerReview"
            ],
        "mask_token" :  "<MASK>"
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # collate = ItemTextCollate(tokenizer, max_length=256)
    # train_dataset = ItemTextDataset("output.json")

    collate = Simmc2Collate(tokenizer, max_length=256)
    # train_dataset = Simmc2Dataset(tokenizer, "/home/yschoi/SIMMC2/data/retrieval/train.json")    
    # sampler = SequentialSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=1, collate_fn=collate, num_workers=1)
    
    eval_dataset = EvalDataset(tokenizer, "/home/yschoi/SIMMC2/retrieval/postprocessing_data.json", )
    sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=sampler, batch_size=1, num_workers=1)
    for batch in eval_dataloader:
        print(batch.keys())
        # inputs, labels = mask_tokens(batch["encoder_input_ids"], tokenizer, 0.15)
        ipdb.set_trace()
