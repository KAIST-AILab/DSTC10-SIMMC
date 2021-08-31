import os
import uuid
import datetime
import warnings

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from common.utils import get_parser
from baseline.modules import Baseline
from baseline.dataset import BaselineDataModule
from baseline.options import TrainingOptions

warnings.simplefilter("ignore")
os.environ['PYTHONWARNINGS'] = "ignore"

if __name__ == "__main__":
    parser = get_parser(TrainingOptions)
    args = parser.parse_args()
    
    # Add experiment ids
    args.exp_uuid4 = str(uuid.uuid4()).split('-')[-1]
    args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    loggers = [
        TensorBoardLogger(
            save_dir=args.log_path,
            name="{}_{}".format(args.timestamp, args.exp_uuid4)
        )
    ]

    if not args.seed < 0:   
        pl.seed_everything(args.seed, workers=True)

    # Checkpointing callback
    checkpoint_callback = ModelCheckpoint(
        monitor="devtest_mean_perp",
        dirpath="./baseline/checkpoints",
        save_weights_only=False,
        # every_n_train_steps=500,
        filename="{step}-{dev_mean_perp:.4f}-{devtest_mean_perp:.4f}",
        save_top_k=3,
        mode='min'
    )
    learning_rate_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=False
    )

    # Distrubted training plugin
    distributed_plugin = None
    if torch.cuda.device_count() > 1:
        distributed_plugin = "dp"
    if args.ddp:
        distributed_plugin = "ddp"

    # Trainer
    trainer_args = \
        {
            "accelerator": distributed_plugin,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "amp_backend": "native",
            "auto_lr_find": True,
            "callbacks": [checkpoint_callback, learning_rate_monitor],
            "deterministic": args.deterministic,
            "gpus": None if args.debug else args.gpus,
            "logger": loggers,
            "log_every_n_steps": args.log_interval,
            "max_epochs": args.max_epochs,
            "plugins": DDPPlugin(find_unused_parameters=False) if args.ddp else None,
            "precision": 16 if args.fp16 else 32,
            "progress_bar_refresh_rate": None if args.do_test else 0,
            "resume_from_checkpoint": args.checkpoint,
            "val_check_interval": args.val_check_interval
        } 
    trainer = pl.Trainer(**trainer_args)

    # Data module (prepare_data, setup called before model init. -- we use tokenizer info for model init.)
    pl_data = BaselineDataModule(args)
    pl_data.prepare_data()

    if args.checkpoint:
        pl_model = Baseline.load_from_checkpoint(
            args.checkpoint,
            args=args, tokenizer=pl_data.tokenizer
        )
    else:
        pl_model = Baseline(args, pl_data.tokenizer)

    if args.do_tune:
        pl_data.setup('fit')
        trainer.tune(model=pl_model, datamodule=pl_data)

    if args.do_train:
        pl_data.setup('fit')
        trainer.fit(model=pl_model, datamodule=pl_data)

    if args.do_test:
        pl_data.setup('test')
        trainer.test(model=pl_model, datamodule=pl_data)
