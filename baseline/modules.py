from typing import Optional, Dict
from argparse import Namespace
from packaging.version import parse

import torch

from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from torchmetrics import (
    MetricCollection,
    AverageMeter
)
from torchmetrics.metric import Metric
from transformers import (
    AutoModelWithLMHead,
    BatchEncoding,
    get_linear_schedule_with_warmup
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from common.utils import get_logger, get_args_string
from common.metrics import DSTScore, DisambAccuracy, BLEUScore
from baseline.process import clean_tokens, parse_dst, parse_response

class Baseline(LightningModule):
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
    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizerBase):
        super().__init__()

        self.args = args

        # Console logger
        self.console_logger = get_logger(
            args.log_path, args.exp_uuid4, args.timestamp
        )
        self.console_log(get_args_string(self.args))
        
        # Model and Tokenizer (tokenizer is initialized in data module)
        self.tokenizer = tokenizer
        self.model = AutoModelWithLMHead.from_pretrained(
            args.config_name, pad_token_id=self.tokenizer.pad_token_id
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Objective (ignore pad token index for efficiency)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="none"
        )

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

        if stage in ("fit", "validate"):
            metrics = {
                "mean_loss": AverageMeter(
                    compute_on_step=False,
                    dist_sync_on_step=True
                )
            }
            self.dev_metrics = MetricCollection(metrics, prefix="dev_")
            self.devtest_metrics = MetricCollection(metrics, prefix="devtest_")
        elif stage in ("test", "predict"):
            self.disambiguate_metrics = DisambAccuracy(
                compute_on_step=False,
                dist_sync_on_step=True
            )
            self.dst_metrics = DSTScore(
                compute_on_step=False,
                dist_sync_on_step=True
            )
            self.generation_metrics = BLEUScore(
                compute_on_step=False,
                dist_sync_on_step=True
            )
        
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
        # Position id hack is already implemented in transformers (master)
        output = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].shape[1] + 100,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.args.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        losses, _ = self(inputs, labels['input_ids'])
        losses = losses.mean()
        perplx = losses.exp()
        if not self.global_step % self.args.log_interval:
            self.console_log(
                "[{} / {}] train loss: {:.5f} perp.: {:.3f}".format(
                    self.global_step, self.num_training_steps-1,
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
        losses, _ = self(inputs, labels['input_ids'])
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
        self.console_log(
            "dev perp.: {:.4f} devtest perp. {:.4f}".format(
                dev_metrics['dev_mean_perp'], 
                devtest_metrics['devtest_mean_perp']
            )
        )
        self.log_dict(dev_metrics)
        self.log_dict(devtest_metrics)

    def test_step(self, batch, batch_idx):
        inputs, labels, disamb_labels = batch
        output = self.generate(inputs)

        batch_ground_truths = clean_tokens(
            self.tokenizer.batch_decode(
                labels['input_ids'], clean_up_tokenization_spaces=True
            )
        )
        batch_predictions = clean_tokens(
            self.tokenizer.batch_decode(
                output, clean_up_tokenization_spaces=True
            )
        )
        for ground_truth, prediction, disamb_label in zip(batch_ground_truths, batch_predictions, disamb_labels):
            # DST parsing
            ground_truth_parsed = parse_dst(ground_truth)
            prediction_parsed = parse_dst(prediction)
            # Response parsing
            ground_truth_response = parse_response(ground_truth)
            prediction_response = parse_response(prediction)

            # Log generated sequences
            self.console_log("{} <==> {} ".format(prediction, ground_truth))
            # Metric tracking
            self.disambiguate_metrics(prediction_parsed, disamb_label)
            self.dst_metrics(prediction_parsed, ground_truth_parsed)
            self.generation_metrics(ground_truth_response, prediction_response)

    def test_epoch_end(self, step_outputs):
        disambiguate_results = self.disambiguate_metrics.compute()
        self.disambiguate_metrics.reset()

        dst_results = self.dst_metrics.compute()
        self.dst_metrics.reset()

        generation_results = self.generation_metrics.compute()
        self.generation_metrics.reset()

        # Disambiguation
        self.console_log("********** Disambiguation Evaluation Results **********")
        self.console_log("disambiguation acc = {}".format(disambiguate_results))
        self.log("disambiguation_acc", disambiguate_results)
        # Dialog State Tracking
        self.console_log("********** DST Evaluation Results **********")
        for k, v in dst_results.items():
            self.console_log("{} = {}".format(k, v.item() if isinstance(v, torch.Tensor) else v))
        self.log_dict(dst_results)
        # Response Generation
        self.console_log("********** Response Generation Results **********")
        for k, v in generation_results.items():
            self.console_log("{} = {}".format(k, v.item() if isinstance(v, torch.Tensor) else v))
        self.log_dict(generation_results)
