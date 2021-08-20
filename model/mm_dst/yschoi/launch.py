import os
import json
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
from backbone import NestedTensor

import argparse
from parse import parse
from backbone import build_backbone_test
from ys_model import Model
from dataset import SceneDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

import ipdb
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    args = parse(parser)       
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    experiment = "exp_" + args.task_type 
    if args.with_image:
        experiment += "_with_img"
    experiment += f"_lr_{args.learning_rate}"
    
    writer = SummaryWriter(os.path.join(args.summary_dir,experiment))
    print(f"SummaryWriter Directory : {os.path.join(args.summary_dir,experiment)}")
    
    os.makedirs(args.model_checkpoint_dir+f"/{experiment}", exist_ok=True)
    print(f"Model checkpoint Directory : {os.path.join(args.model_checkpoint_dir,experiment)}")
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, cache_dir=None
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, cache_dir=None
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError(
                "Additional special tokens file {args.add_special_tokens} not found}"
            )
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")
        
        vocabs = tokenizer.get_vocab()
        vocab_size = len(tokenizer)
        for k, v in vocabs.items():
            if v in range(10):
                print(f"{v}: {k}")

    # Data Loader
    train_dataset = SceneDataset(tokenizer, None, file_path="/ext/dstc10/yschoi/train.json", image_root_dir="/ext/coco_dataset/simmc2/data")
    
    def collate_bart(examples: Dict):
        predict = list(map(lambda x: x['predict'], examples)) # List[str]
        belief = list(map(lambda x: x['belief'], examples)) # List[str]
        imgs = NestedTensor.from_tensor_list(list(map(lambda x: x["image"], examples)))
        bbox = list(map(lambda x: x['bbox'], examples))

        assert tokenizer._pad_token is not None
        predict = tokenizer.batch_encode_plus(predict, padding="longest", truncation=True, return_tensors="pt") # Dict
        belief = tokenizer.batch_encode_plus(belief, padding="longest", truncation=True, return_tensors="pt").input_ids #
        bbox_pad = pad_sequence(bbox, batch_first=True) # padding space occupied with 0
        return {
            "predict" : predict,
            "belief" : belief,
            "image": imgs,
            "bbox" : bbox_pad,
        }

    sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=4, collate_fn=collate_bart, num_workers=4)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_bart, num_workers=4)
    
    # Model
    backbone = build_backbone_test(args)
    model = Model(args, vocab_size, backbone)
    model.to(device)

    # Optimizer        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    t_total = (len(train_dataloader)* args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # Train Start
    model.to(device)
    model.train()
    model.zero_grad()
    _step = 0
    epochs_trained = 0
    for epoch in range(args.num_train_epochs):
        for batch in tqdm(train_dataloader, desc="Training"):
            
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            labels = batch["belief"][:, 1:].contiguous() # bs, seq_len-1
            logits = model(batch)[:, :-1, :].contiguous() # bs, seq_len-1, vocab_size
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            if _step % 50 == 0:
                print(f"Loss {loss.item()}")
                writer.add_scalar("train/loss", loss.item(), global_step=_step)
            # save model
            if _step % args.save_freq == 0:
                print("============SAVE Model============")
                torch.save(model.state_dict(), os.path.join(args.model_checkpoint_dir, f"{experiment}/step_{_step}.pkl"))
            _step += 1        
        print(f"Epoch {epoch} Done !")


