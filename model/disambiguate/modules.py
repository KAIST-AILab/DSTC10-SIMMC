import json

from typing import Optional, Dict
from argparse import Namespace

import torch

from torch import nn

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only
from torchmetrics import AverageMeter
from transformers import (
    BatchEncoding,
    get_linear_schedule_with_warmup
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import RobertaForSequenceClassification

from metrics import DisambiguateAccuracy


class DisambiguationModel(LightningModule):

    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.args = args
        
        # Model and Tokenizer (tokenizer is initialized in data module)
        self.tokenizer = tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(
            args.pretrained_checkpoint, num_labels=2
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.config = self.model.config

        # Objective
        self.criterion = nn.CrossEntropyLoss(reduce="none")

        # Save hparams
        self.save_hyperparameters(vars(args))      

    @rank_zero_only
    def console_log(self, text: str):
        print(text)

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit":
            num_gpus = self.trainer.num_gpus
            num_nodes = self.trainer.num_nodes
            total_devices = max(num_gpus * num_nodes, 1)
            train_batches = len(self.train_dataloader()) // total_devices
            self.num_training_steps = (self.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches

        if stage in ("fit", "validate"):
            self.train_mean_loss = AverageMeter(
                compute_on_step=True,
                dist_sync_on_step=True
            )
            self.dev_mean_loss = AverageMeter(
                compute_on_step=False,
                dist_sync_on_step=True
            )
            self.dev_disambiguate_acc = DisambiguateAccuracy(
                compute_on_step=False,
                dist_sync_on_step=True
            )
            self.devtest_mean_loss = AverageMeter(
                compute_on_step=False,
                dist_sync_on_step=True
            )
            self.devtest_disambiguate_acc = DisambiguateAccuracy(
                compute_on_step=False,
                dist_sync_on_step=True
            )
        elif stage in ("test", "predict"):
            self.disambiguation_predicted = list()
            self.devtest_disambiguate_acc = DisambiguateAccuracy(
                compute_on_step=False,
                dist_sync_on_step=True
            )
        else:
            raise NotImplemented
        
    def configure_optimizers(self) -> Dict:
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
        # Get scheduler
        num_warmup_steps = int(self.num_training_steps * self.args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, self.num_training_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def forward(
        self, source: BatchEncoding, labels: Optional[torch.Tensor]=None
    ):
        output = self.model(**source)
        logits = output.logits
        losses = self.criterion(logits, labels)
        return logits, losses

    def training_step(self, batch, batch_idx):
        _, _, source, labels = batch
        _, losses = self(source, labels)
        mean_loss = losses.mean()
        if not self.global_step % self.args.log_interval:
            self.console_log(
                "[{} / {}] train loss: {:.5f}".format(
                    self.global_step, self.num_training_steps-1,
                    mean_loss.item()
                )
            )
        return mean_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        _, _, source, labels = batch
        logits, losses = self(source, labels)

        # dev
        if dataloader_idx == 0:
            self.dev_mean_loss(losses)
            self.dev_disambiguate_acc(logits, labels)
        # devtest
        else:
            self.devtest_mean_loss(losses)
            self.devtest_disambiguate_acc(logits, labels)
        
    def validation_epoch_end(self, step_outputs):
        dev_disambiguate_acc = self.dev_disambiguate_acc.compute()
        devtest_disambiguate_acc = self.devtest_disambiguate_acc.compute()

        self.dev_mean_loss.reset()
        self.dev_disambiguate_acc.reset()
        self.devtest_mean_loss.reset()
        self.devtest_disambiguate_acc.reset()

        self.console_log(
            "dev acc.: {:.4f} / devtest acc.: {:.4f}".format(
                dev_disambiguate_acc,
                devtest_disambiguate_acc,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
        )

    def test_step(self, batch, batch_idx):
        (
            dialog_idx,
            turn_id,
            inputs, 
            disamb_labels,
        ) = batch
        cl_logits = self(inputs)

        for di, ti, lg in zip(dialog_idx, turn_id, cl_logits.max(dim=-1)[1].tolist()):
            self.disambiguation_predicted.append(
                {
                    'dialog_idx': di,
                    'turn_idx': ti,
                    'predicted': lg
                }
            )
        self.devtest_disambiguate_acc(cl_logits, disamb_labels)

    def test_epoch_end(self, step_outputs):
        disambiguate_results = self.devtest_disambiguate_acc.compute()
        self.devtest_disambiguate_acc.reset()
        self.console_log(
            "disamb. acc. (devtest): {:.4f}".format(disambiguate_results)
        )

        with open(self.args.predicted_output_path, 'w') as f:
            json.dump(self.disambiguation_predicted, "output.json")

