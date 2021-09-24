from typing import Optional, Dict
from argparse import Namespace
from packaging.version import parse

import torch

from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from torchmetrics import AverageMeter
from transformers import (
    AutoConfig,
    BatchEncoding,
    get_linear_schedule_with_warmup
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils import get_logger, get_args_string
from modeling import BartForMultiTask
from process import clean_tokens
from metrics import DisambiguateAccuracy


class MultiTask(LightningModule):
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
        # self.model_config = AutoConfig.from_pretrained(
        #     args.config_name, num_labels=2
        # )
        self.model = BartForMultiTask.from_pretrained(
            args.config_name, num_classes=2,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Objective (ignore pad token index for efficiency)
        self.criterion = lambda l1, l2: l1 + args.mix_lambda * l2

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
            self.predicted = list()
            self.targets = list()

            self.devtest_disambiguate_acc = DisambiguateAccuracy(
                compute_on_step=False,
                dist_sync_on_step=True
            )
            # self.devtest_disambiguate_acc = SystemDisambiguateAccuracy(
            #     compute_on_step=False,
            #     dist_sync_on_step=True
            # )
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
        self,
        source: BatchEncoding,
        target: Optional[torch.Tensor],
        cl_labels: Optional[torch.Tensor]=None
    ):
        output = self.model(
            input_ids=source['input_ids'],
            attention_mask=source['attention_mask'],
            decoder_input_ids=target['input_ids'][:,:-1],
            decoder_attention_mask=target['attention_mask'][:,:-1],
            lm_labels=target['input_ids'][:,1:],
            cl_labels=cl_labels,
            is_generation=False
        )
        return output

    def generate(self, inputs: BatchEncoding):
        generated = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].shape[1] + 160,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.args.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id
        )
        cl_logits = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            is_generation=False
        ).cl_logits
        return generated, cl_logits

    def training_step(self, batch, batch_idx):
        source, target, cl_labels = batch
        output = self(source, target, cl_labels)

        mean_lm_loss = output.loss.mean()
        mean_perplex = mean_lm_loss.exp()
        mean_cl_loss = output.cl_loss
        total_losses = self.criterion(mean_lm_loss, mean_cl_loss)
        
        if not self.global_step % self.args.log_interval:
            self.console_log(
                "[{} / {}] train loss: {:.5f} perp.: {:.3f}".format(
                    self.global_step, self.num_training_steps-1,
                    total_losses.item(), mean_perplex.item()
                )
            )
        self.log(
            "train_loss", total_losses, 
            on_step=True, on_epoch=True, sync_dist=True, logger=True 
        )
        self.log(
            "train_cl_loss", mean_cl_loss,
            on_step=True, on_epoch=True, sync_dist=True, logger=True 
        )
        self.log(
            "train_lm_loss", mean_lm_loss,
            on_step=True, on_epoch=True, sync_dist=True, logger=True 
        )
        self.log(
            "train_perp", mean_perplex,
            on_step=True, on_epoch=True, sync_dist=True, logger=True
        )
        return total_losses

    def validation_step(self, batch, batch_idx, dataloader_idx):
        source, target, cl_labels = batch
        output = self(source, target, cl_labels)

        cl_logits = output.cl_logits
        mean_lm_loss = output.loss.mean()
        mean_perplex = mean_lm_loss.exp()
        mean_cl_loss = output.cl_loss
        total_losses = self.criterion(mean_lm_loss, mean_cl_loss)

        # dev
        if dataloader_idx == 0:
            self.dev_mean_loss(total_losses)
            self.dev_disambiguate_acc(cl_logits, cl_labels)
            self.log(
                "dev_loss", total_losses,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
            self.log(
                "dev_perp", mean_perplex,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
        # devtest
        else:
            self.devtest_mean_loss(total_losses)
            self.devtest_disambiguate_acc(cl_logits, cl_labels)
            self.log(
                "devtest_loss", total_losses,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
            self.log(
                "devtest_perp", mean_perplex,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )

    def validation_epoch_end(self, step_outputs):
        dev_mean_loss = self.dev_mean_loss.compute()
        devtest_mean_loss = self.devtest_mean_loss.compute()
        dev_disambiguate_acc = self.dev_disambiguate_acc.compute()
        devtest_disambiguate_acc = self.devtest_disambiguate_acc.compute()

        self.dev_mean_loss.reset()
        self.dev_disambiguate_acc.reset()
        self.devtest_mean_loss.reset()
        self.devtest_disambiguate_acc.reset()
        
        dev_mean_perplx = dev_mean_loss.exp()
        devtest_mean_perplx = devtest_mean_loss.exp()
        self.console_log(
            "dev perp.: {:.4f} acc.: {:.4f} / devtest perp. {:.4f} acc.: {:.4f}".format(
                dev_mean_perplx.item(),
                dev_disambiguate_acc,
                devtest_mean_perplx.item(),
                devtest_disambiguate_acc,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )
        )
        self.log(
            "dev_mean_loss", dev_mean_loss,
            on_step=False, on_epoch=True, sync_dist=True, logger=True
        )
        self.log(
            "dev_disambiguate_acc", dev_disambiguate_acc,
            on_step=False, on_epoch=True, sync_dist=True, logger=True
        )
        self.log(
            "devtest_mean_loss", devtest_mean_loss,
            on_step=False, on_epoch=True, sync_dist=True, logger=True
        )
        self.log(
            "devtest_disambiguate_acc", devtest_disambiguate_acc,
            on_step=False, on_epoch=True, sync_dist=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        inputs, labels, disamb_labels = batch
        generated, cl_logits = self.generate(inputs)

        batch_ground_truths = clean_tokens(
            self.tokenizer.batch_decode(
                labels['input_ids'], clean_up_tokenization_spaces=True
            ), self.tokenizer.pad_token, self.tokenizer.bos_token,
            self.tokenizer.eos_token
        )
        
        batch_predictions = clean_tokens(
            self.tokenizer.batch_decode(
                generated, clean_up_tokenization_spaces=True
            ), self.tokenizer.pad_token, self.tokenizer.bos_token,
            self.tokenizer.eos_token
        )

        self.console_log(
            "{}\n{}".format(batch_ground_truths[0], batch_predictions[0])
        )
        self.predicted.extend(batch_predictions)
        self.targets.extend(batch_ground_truths)        

        self.devtest_disambiguate_acc(cl_logits, disamb_labels)

    def test_epoch_end(self, step_outputs):
        disambiguate_results = self.devtest_disambiguate_acc.compute()
        self.devtest_disambiguate_acc.reset()

        self.console_log(
            "Disambiguation Accuracy : {:.4f}".format(disambiguate_results)
        )

        with open(self.args.predicted_output_path, 'w') as f:
            for line in self.predicted:
                f.write("{}\n".format(line.strip()))
            
        with open(self.args.target_output_path, 'w') as f:
            for line in self.targets:
                f.write("{}\n".format(line.strip()))
