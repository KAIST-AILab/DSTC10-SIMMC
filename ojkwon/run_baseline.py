import uuid
import datetime

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from common.utils import get_parser
from baseline.modules import Baseline
from baseline.dataset import BaselineDataModule
from baseline.options import TrainingOptions
 
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
        dirpath="./checkpoints/baseline",
        every_n_train_steps=0,
        every_n_val_epochs=1,
        filename="{epoch}-{devtest_mean_perp:.3f}",
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
            "precision": 16 if args.fp16 else 32,
            "progress_bar_refresh_rate": 0,
            "resume_from_checkpoint": args.checkpoint,
        } 
    trainer = pl.Trainer(**trainer_args)
    args.num_gpus = trainer.num_gpus
    args.num_nodes = trainer.num_nodes

    # Data module (prepare_data, setup called before model init. -- we use tokenizer info for model init.)
    pl_data  = BaselineDataModule(args)
    pl_data.prepare_data()
    pl_data.setup()
    # Model
    pl_model = Baseline(args, pl_data.tokenizer)

    if args.do_tune:    
        trainer.tune(model=pl_model, datamodule=pl_data)

    if args.do_train:
        trainer.fit(model=pl_model, datamodule=pl_data)

    if args.do_test:
        trainer.test(model=pl_model, datamodule=pl_data)
