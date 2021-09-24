import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint

from utils import get_parser 
from options import TrainingOptions
from dataset import DisambiguationDataModule
from modules import DisambiguationModel

if __name__ == "__main__":

    parser = get_parser(TrainingOptions)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor="devtest_acc",
        dirpath="./checkpoints",
        save_weights_only=False,
        # every_n_train_steps=500,
        filename="{step}-{devtest_acc:.4f}",
        save_top_k=3,
        mode='min'
    )
    distributed_plugin = None
    if torch.cuda.device_count() > 1:
        distributed_plugin = "dp"
    if args.ddp:
        distributed_plugin = "ddp"


    trainer_args = \
        {
            "accelerator": distributed_plugin,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "amp_backend": "native",
            "auto_lr_find": True,
            "callbacks": [checkpoint_callback],
            "deterministic": args.deterministic,
            "gpus": args.gpus,
            "log_every_n_steps": 100,
            "max_epochs": args.max_epochs,
            "precision": 16 if args.fp16 else 32,
            "progress_bar_refresh_rate": None if args.do_test else 0,
            "resume_from_checkpoint": args.checkpoint,
            "val_check_interval": args.val_check_interval
        }
    trainer = pl.Trainer(**trainer_args)

    pl_data = DisambiguationDataModule(args)
    pl_data.prepare_data()
    # pl_data.setup()

    if args.checkpoint:
        pl_model = DisambiguationModel.load_from_checkpoint(
            args.checkpoint,
            args=args, tokenizer=pl_data.tokenizer
        )
    else:
        pl_model = DisambiguationModel(args, pl_data.tokenizer)

    if args.do_train:
        pl_data.setup('fit')
        trainer.fit(model=pl_model, datamodule=pl_data)


    if args.do_test:
        pl_data.setup('test')
        trainer.test(model=pl_model, datamodule=pl_data)

        pl_model.model.save_pretrained('best-checkopint-disamb')
        pl_model.tokenizer.save_pretrained('best-checkopint-disamb')

