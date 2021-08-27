from logging import logProcesses
from typing import Optional
from argparse import Namespace

import torch

from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from torchmetrics import (
    MetricCollection,
    AverageMeter
)
from transformers import (
    AutoModelWithLMHead,
    BatchEncoding,
    get_linear_schedule_with_warmup
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from common.utils import get_logger, get_args_string
from baseline.process import batch_remove_pad_token

class Baseline(LightningModule):
    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.args = args

        # Console logger
        self.console_logger = get_logger(
            args.log_path, args.exp_uuid4, args.timestamp
        )
        self.console_log(get_args_string(self.args))
        

        # Model and Tokenizer (tokenizer is initialized in data module)
        self.lr = args.learning_rate
        self.model = AutoModelWithLMHead.from_pretrained(args.config_name)

        self.tokenizer = tokenizer
        
        # GPT-2 does not have a pad token, so reset embedding like so
        self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is not None:
            old_embed = self.model.get_input_embeddings()
            old_embed.weight.data[self.tokenizer.pad_token_id].zero_()

        # Objective (ignore pad token index for efficiency)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="none"
        )
        
        # Metric monitors
        metrics = {
            "mean_loss": AverageMeter(
                compute_on_step=False,
                dist_sync_on_step=True
            )
        }
        self.dev_metrics = MetricCollection(metrics, prefix="dev_")
        self.devtest_metrics = MetricCollection(metrics, prefix="devtest_")

        # Save hparams
        self.save_hyperparameters(vars(args))
    
    def setup(self, stage):
        if stage == 'fit':
            total_devices = self.args.num_gpus * self.args.num_nodes
            if total_devices == 0:
                total_devices = 1
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.args.max_epochs * train_batches) // self.args.accumulate_grad_batches

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n,p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n,p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(
            grouped_params, lr=self.args.learning_rate,
            betas=(self.args.beta_1, self.args.beta_2),
            weight_decay=self.args.weight_decay
        )
        num_warmup_steps = int(self.train_steps * self.args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, self.train_steps)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @rank_zero_only
    def console_log(self, text: str):
        self.console_logger.info(text)

    def forward(
        self,
        inputs: BatchEncoding,
        labels: Optional[torch.Tensor]=None
    ):
        logits = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        ).logits
        # Autoregressive LM loss (modified for pad tokens)
        losses = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            losses = self.criterion(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1)
            )
        return (losses, logits) if losses is not None else logits
    
    def generate(self, inputs: BatchEncoding):
        output = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.args.num_return_sequences,
        )
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        losses, _ = self(inputs, labels)
        losses = losses.mean()
        perplx = losses.exp()
        if not self.global_step % self.args.log_interval:
            self.console_log(
                "[{} / {}] Train Loss.: {:.5f} Perp.: {:.3f}".format(
                    self.global_step, self.train_steps-1,
                    losses.item(), perplx.item()
                )
            )
        self.log(
            "train_loss", losses, 
            on_step=True, on_epoch=True, sync_dist=True, logger=True 
        )
        self.log(
            "train_perp", perplx,
            on_step=True, on_epoch=True, sync_dist=True, logger=True
        )
        return losses

    def validation_step(self, batch, batch_idx, dataloader_idx):
        inputs, labels = batch
        losses, _ = self(inputs, labels)
        # dev
        if dataloader_idx == 0:
            self.dev_metrics(losses)
            losses = losses.mean()
            perplx = losses.exp()
            self.log(
                "dev_loss", losses,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
            self.log(
                "dev_perp", perplx,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
        # devtest
        else:
            self.devtest_metrics(losses)
            losses = losses.mean()
            perplx = losses.exp()
            self.log(
                "devtest_loss", losses,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
            self.log(
                "devtest_perp", perplx,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )            

    def validation_epoch_end(self, step_outputs):
        dev_metrics = self.dev_metrics.compute()
        devtest_metrics = self.devtest_metrics.compute()
        self.dev_metrics.reset()
        self.devtest_metrics.reset()
        
        dev_metrics['dev_mean_perp'] = dev_metrics['dev_mean_loss'].exp()
        devtest_metrics['devtest_mean_perp'] = devtest_metrics['devtest_mean_loss'].exp()
        self.log_dict(dev_metrics)
        self.log_dict(devtest_metrics)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self.generate(inputs)

        ground_truth = batch_remove_pad_token(
            self.tokenizer.batch_decode(labels)
        )
        predicted = batch_remove_pad_token(
            self.tokenizer.batch_decode(output)
        )
        for g, p in zip(ground_truth, predicted):
            self.console_log("{} <=> {}".format(g, p))
        
