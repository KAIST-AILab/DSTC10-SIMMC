from typing import Optional, Dict
from argparse import Namespace

import torch
import numpy as np

from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from torchmetrics import (
    MetricCollection,
    AverageMeter
)
from transformers import (
    AutoModel,
    PreTrainedTokenizerBase,
    BatchEncoding,
    get_linear_schedule_with_warmup
)
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from clip.modeling import SimpleCLIP
from common.utils import get_logger, get_args_string

class CLIP(LightningModule):
    '''
        GPT-2 end-to-end baseline for SIMMC 2.0 dataset provided by the organizers.

        Args:
            args <Namespace>: arguments
            tokenizer <PreTrainedTokenizerBase>: tokenizer

        Attributes:
            console_logger <Logger>: logger for CLI output
            model <nn.Module>: neural model
            criterion <Union[Callable, nn.Module]>: loss function
    '''
    def __init__(self, args: Namespace, tokenizer: Optional[PreTrainedTokenizerBase]=None):
        super().__init__()

        self.model = SimpleCLIP(args)

        self.args = args
        # self.tokenizer = tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer

        # Console logger
        self.console_logger = get_logger(
            args.log_path, args.exp_uuid4, args.timestamp
        )
        self.console_log(get_args_string(self.args))
        
        # GPT-2 does not have a pad token, so reset embedding like so
        # self.model.resize_token_embeddings(len(self.tokenizer))

        # Save hparams
        self.save_hyperparameters(vars(args))

    @rank_zero_only
    def console_log(self, text: str):
        self.console_logger.info(text)

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit":
            num_gpus = self.trainer.num_gpus
            num_nodes = self.trainer.num_nodes
            total_devices = max(num_gpus * num_nodes, 1)
            train_batches = len(self.train_dataloader()) // total_devices
            self.num_training_steps = (self.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches

        if stage == 'test':
            self.object_ids = list()
            self.object_embeddings = list()
            self.metadata_embeddings = list()
        
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

    def forward(self, objects: torch.Tensor, metadata: torch.Tensor):
        return self.model(objects, metadata)
    
    def training_step(self, batch, batch_idx):
        objects, metadata = batch
        outputs = self(objects, metadata)

        if not self.global_step % self.args.log_interval:
            self.console_log(
                "[{} / {}] train loss.: {:.5f}, item: {:.4f}, metadata: {:.4f}".format(
                    self.global_step, self.num_training_steps-1,
                    outputs.total_loss.item(),
                    outputs.metadata_loss.item(),
                    outputs.object_loss.item()
                )
            )
        return outputs.total_loss

    def test_step(self, batch, batch_idx):
        objects, metadata = batch
        outputs = self(objects, metadata)

        self.object_ids.extend(objects.cpu().numpy().tolist())
        self.object_embeddings.extend(outputs.object_embedding.cpu().numpy().tolist())
        self.metadata_embeddings.extend(outputs.metadata_embedding.cpu().numpy().tolist())

    def test_epoch_end(self, outputs) -> None:
        tsne = TSNE(
            n_components=2
        ).fit_transform(np.array(self.object_embeddings))

        dictionary = {"fashion": "tab:blue", "furniture": "tab:orange"}

        fig, ax = plt.subplots()
        plt.figure(figsize=(150,150))
        for item_id, embed in zip(self.object_ids, tsne):
            # furniture domain starts at 289
            if item_id < 289:
                color = dictionary['fashion']
            else:
                color = dictionary['furniture']
            ax.scatter(embed[0], embed[1], c=color, s=10)
            ax.annotate(item_id, (embed[0], embed[1]), fontsize='xx-small')
        markers = [
            plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dictionary.values()
        ]
        ax.legend(markers, dictionary.keys(), numpoints=1)
        ax.set_box_aspect(1)

        fig.savefig(
            self.args.image_path, dpi=600
        )
        