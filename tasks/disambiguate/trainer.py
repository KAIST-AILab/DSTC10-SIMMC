import argparse
import os
import sys
import time
import random
import datetime

import attr
import json

import torch
import numpy as np

from math import ceil
from argparse import Namespace

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp
from torch import distributed as dist
from torch.optim import AdamW

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)

from options import get_parser, get_logger
from loader import DisambiguationCollector, DisambiguationDataset

class ConditionalContext:
    """
        Context mangager wrapper that enters only if the condition is true.
        Inlcuded here for better readability.

        ex. if fp16:
                with torch.cuda.amp.autocast():
                    y = model(x)
            else:
                y = model(x)
        
        ->
            with ContditionalContext(fp16, torch.cuda.amp.autocast()):
                y = model(x)

        Args:
            conditional <bool> : conditional statement
            contextmanager <ContextManager> : context manager
    """
    def __init__(self, condition: bool, contextmanager):
        self.condition = condition
        self.contextmanager = contextmanager

    def __enter__(self):
        if self.condition:
            return self.contextmanager.__enter__()

    def __exit__(self, *args):
        if self.condition:
            return self.contextmanager.__exit__(*args)

@attr.s
class Trainer:

    args: Namespace = attr.ib()

    def __attrs_post_init__(self):
        args = self.args

        self.distributed = (args.local_rank != -1)
        self.main_process = (args.local_rank in (0, -1))

        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")

        if not os.path.isdir("./log"):
            os.mkdir("./log")
    
    def log(self, info):
        if self.args.local_rank in (0, -1):
            self.logger.info(str(info))
            
    def write(self, func, args):
        if self.args.local_rank in (0, -1):
            getattr(self.writer, func)(**args)

    def log_args(self):
        self.log("==================== Arguments ====================")
        for k, v in sorted(vars(self.args).items()):
            k_str = "{}".format(k) + (" " * (30 - len(k)))
            self.log("{} : {}".format(k_str, v))
        self.log("===================================================")

    def fix_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def train(self, *args):
        args = self.args
        device = "cuda"

        if args.seed >= 0:
            self.fix_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.config_name)
        
        # If DDP, init process group and set device
        if self.distributed:
            dist.init_process_group(
                backend='nccl',
                rank=args.local_rank,
                world_size=args.world_size
            )
            device = "cuda:{}".format(args.local_rank)

        # If main process, instantiate logger and writer
        if self.main_process:
            self.logger = get_logger(args)
            self.log("Intializing Trainer ...")
            self.log_args()
            self.writer = SummaryWriter("tb_log/disambiguate/{}_{}".format(
                args.config_name,
                datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            ))

        # Init tokenizer
        model_kwargs = {
            "num_labels": 2,
            "hidden_dropout_prob": args.dropout
        }
        if "xlnet" in args.config_name:
            model_kwargs.pop("hidden_dropout_prob")
            model_kwargs["dropout"] = args.dropout

        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.config_name,
            **model_kwargs
        )
        if args.use_special_tokens:
            num_added_tokens = tokenizer.add_special_tokens(
                {"additional_special_tokens": ["[USER]", "[SYS]"]}
            )
            self.model.resize_token_embeddings(len(tokenizer))

        if self.distributed:
            torch.cuda.set_device(args.local_rank)
        self.model.to(device)
        
        # Wrap model for DDP
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                find_unused_parameters=False
            )

        # Init optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params= [
            {
                "params": [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        # grouped_params = self .model.parameters()
        optim_cls = AdamW
        optim_arg = {
            "lr": args.learning_rate,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay
        }
        self.optimizer = optim_cls(params=grouped_params, **optim_arg)
        if args.sharded:
            self.optimizer = OSS(params=grouped_params, optim=optim_cls, **optim_arg)
        
        collate_fn = DisambiguationCollector(args, tokenizer)
        sep_token = tokenizer.sep_token

        # Init dataset
        train_dataset = DisambiguationDataset(args, args.train_file, sep_token)
        train_sampler = DistributedSampler(train_dataset) if self.distributed else RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.workers,
            pin_memory=True
        )
        dev_dataset = DisambiguationDataset(args, args.dev_file, sep_token)
        dev_sampler = DistributedSampler(dev_dataset) if self.distributed else RandomSampler(dev_dataset)
        dev_loader = DataLoader(
            dev_dataset,
            sampler=dev_sampler,
            batch_size=args.batch_size * 4,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        devtest_dataset = DisambiguationDataset(args, args.devtest_file, sep_token)
        devtest_sampler = DistributedSampler(devtest_dataset) if self.distributed else RandomSampler(devtest_dataset)
        devtest_loader = DataLoader(
            devtest_dataset,
            sampler=devtest_sampler,
            batch_size=args.batch_size * 4,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        # Computing steps
        total_steps = len(train_loader) * args.max_epochs
        update_steps_per_epoch = ceil(len(train_loader)  / args.gradient_accumulation_steps)
        if args.max_steps > 0:
            total_steps = args.max_steps
            args.max_epochs = ceil(args.max_steps / update_steps_per_epoch)
        else:
            total_update_steps = args.max_epochs * update_steps_per_epoch
        validate_steps = int(len(train_loader) * args.validate_ratio)

        warmup_steps = args.warmup_steps
        if args.warmup_ratio > 0.:
            warmup_steps = int(total_update_steps * args.warmup_ratio)

        # Init scheduler
        self.scheduler = get_scheduler(
            "linear",
            self.optimizer,
            warmup_steps,
            total_update_steps
        )

        # Init scaler
        if args.fp16:
            self.scaler = GradScaler()

        # criterion = nn.CrossEntropyLoss()
        criterion = lambda x, y: torch.abs(F.cross_entropy(x, y) - args.flooding) + args.flooding

        self.log("Begin Training ...")
        start_time = time.time()
        ckpt_list = list()
        global_step = 1
        best_devtest_acc = -1
        for epoch in range(args.max_epochs):
            # Must shuffle at beginning of each epoch
            if self.distributed:
                dist.barrier()
                train_sampler.set_epoch(epoch)

            self.model.train()
            for batch_idx, (txt, lbl) in enumerate(train_loader, 1):
                txt = {k: v.to(device) for k,v in txt.items()}
                lbl = lbl.to(device)

                # Graident accumulation 
                # If DDP, do not sync until accumulation step
                with ConditionalContext(
                    self.distributed and (global_step % args.gradient_accumulation_steps),
                    self.model.no_sync() if self.distributed else None
                ):
                    # Conditional for FP16
                    with ConditionalContext(args.fp16, autocast()):
                        out = self.model(**txt).logits
                        lss = criterion(out, lbl)
                        lss = lss / args.gradient_accumulation_steps
                        
                    if args.fp16:
                         self.scaler.scale(lss).backward()
                    else:
                        lss.backward()

                # Step
                if not (global_step % args.gradient_accumulation_steps):
                    if args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                item_loss = lss.item()
                if self.distributed:
                    item_loss = self.reduce_mean(lss, args.world_size).item()

                if not (global_step % args.log_steps):
                    self.log("Epoch {} [{} / {}] Loss: {:.5f}".format(
                        epoch,
                        global_step,
                        total_steps,
                        item_loss
                    ))
                    self.write(
                        "add_scalar",
                        {
                            "tag":"train_loss",
                            "scalar_value": item_loss,
                            "global_step": global_step
                        }
                    )

                # Validation
                if not (batch_idx % validate_steps):
                    self.model.eval()
                    self.log("Begin Validation (dev) ...")
                    dev_avg_loss, dev_acc = self.evaluate(args, dev_loader, device)
                    self.log("Epoch {} [{} / {}] Acc. (dev): {:.3f}%".format(
                        epoch,
                        global_step,
                        total_steps,
                        dev_acc * 100
                    ))
                    self.write(
                        "add_scalar",
                        {
                            "tag": "dev_acc",
                            "scalar_value": dev_acc,
                            "global_step": global_step
                        }
                    )
                    self.write(
                        "add_scalar",
                        {
                            "tag": "dev_avg_loss",
                            "scalar_value": dev_avg_loss,
                            "global_step": global_step
                        }
                    )

                    self.log("Begin Evaluation (devtest) ...")
                    devtest_avg_loss, devtest_acc = self.evaluate(args, devtest_loader, device)
                    self.log("Epoch {} [{} / {}] Acc. (devtest): {:.3f}%".format(
                        epoch,
                        global_step,
                        total_steps,
                        devtest_acc * 100
                    ))
                    self.write(
                        "add_scalar",
                        {
                            "tag": "devtest_acc",
                            "scalar_value": devtest_acc,
                            "global_step": global_step
                        }
                    )
                    self.write(
                        "add_scalar",
                        {
                            "tag": "devtest_avg_loss",
                            "scalar_value": devtest_avg_loss,
                            "global_step": global_step
                        }
                    )

                    if devtest_acc > best_devtest_acc:
                        self.log("Best model found ...")
                        best_devtest_acc = devtest_acc

                        if self.distributed:
                            dist.barrier()
                        if self.main_process:
                            ckpt_path = "checkpoint/{}_{}_{:.4f}.ckpt".format(
                                args.config_name,
                                global_step,
                                best_devtest_acc
                            )
                            # Append checkpoint list as tuple 
                            ckpt_list.append((best_devtest_acc, ckpt_path))
                            self.save_model(ckpt_path)

                            while len(ckpt_list) > args.weights_to_keep:
                                ckpt_list = sorted(ckpt_list, reverse=True)
                                to_remove = ckpt_list.pop(-1)[1]
                                os.remove(to_remove)


                    self.model.train()
                
                # Step increment
                global_step += 1

        self.log(
            "Training has ended ... Time elapsed {:.2f} min. Best devtest acc. {:.2f}%".format(
                (time.time() - start_time) / 60.,
                best_devtest_acc * 100
            )
        )

        if self.main_process:
            self.writer.close()
            handlers = self.logger.handlers[:]
            for h in handlers:
                h.close()
                self.logger.removeHandler(h)
    
    @torch.no_grad()
    def save_model(self, checkpoint: str):
        state_dict = {
            "state_dict": None,
            "optimizer": self.optimizer.state_dict()
        }
        if isinstance(self.model, DistributedDataParallel):
            state_dict["state_dict"] = self.model.module.state_dict()
        else:
            state_dict["state_dict"] = self.model.state_dict()
        torch.save(state_dict, checkpoint)
    
    @torch.no_grad()
    def load_model(self, checkpoint: str):
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict["state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    @staticmethod
    def reduce_mean(tensor, world_size):
        t = tensor.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / world_size
        return t

    @torch.no_grad()
    def evaluate(self, args, loader: DataLoader, device):
        args = self.args
        loader = tqdm(loader, leave=False)

        criterion = nn.CrossEntropyLoss()
        crt, tot = 0, 0
        losses = list()
        for txt, lbl in loader:
            txt, lbl = {k: v.to(device) for k,v in txt.items()}, lbl.to(device)
            # Forward pass
            with ConditionalContext(args.fp16, autocast()):
                out = self.model(**txt).logits
                lss = criterion(out, lbl)
            
            item_loss = lss.item()
            _, prd = torch.max(out, dim=-1)
            crt_btc = (prd == lbl).sum()
            tot_btc = torch.Tensor([len(lbl)]).to(device)
            if self.distributed:
                crt_btc = self.reduce_mean(crt_btc, 1)
                tot_btc = self.reduce_mean(tot_btc, 1)
                item_loss = self.reduce_mean(lss, args.world_size).item()
            losses.append(item_loss)        
            crt += crt_btc.item()
            tot += tot_btc.item()
            loader.set_description("Acc.: {:.3f}%".format(crt / tot * 100))
        avg_loss = sum(losses) / len(losses)
        loader.close()
        return avg_loss, crt / tot

            
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

    del trainer
