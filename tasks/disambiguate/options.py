import os
import attr
import logging
import datetime

import torch
import datargs

from argparse import Namespace

@attr.s
class Options:
    train_file: str = attr.ib(
        default="data/disambiguate/simmc2_disambiguate_dstc10_train.json",
        metadata={"help": "path to train file"}
    )
    dev_file: str = attr.ib(
        default="data/disambiguate/simmc2_disambiguate_dstc10_dev.json",
        metadata={"help": "path to dev file"}
    )
    devtest_file: str = attr.ib(
        default="data/disambiguate/simmc2_disambiguate_dstc10_devtest.json",
        metadata={"help": "path to devtest file"}
    )
    sharded: bool = attr.ib(
        default=False,
        metadata={"help": "flag for sharded DDP"}
    )
    log_path: str = attr.ib(
        default="log",
        metadata={"help": "path to logging folder"}
    )
    log_steps: int = attr.ib(
        default=10,
        metadata={"help": "log at every given step"}
    )
    fp16: bool = attr.ib(
        default=False,
        metadata={"help": "flag for automatic mixed precision (fp16)"}
    )
    workers: int = attr.ib(
        default=4,
        metadata={"help": "number of workers for dataloader"}
    )
    local_rank: int = attr.ib(
        default=int(os.environ.get("LOCAL_RANK", -1)),
        metadata={"help": "local rank id"}
    )
    world_size: int = attr.ib(
        default=int(os.environ.get("WORLD_SIZE", torch.cuda.device_count())),
        metadata={"help": "number of procs (nprocs) in this group (usually num. of gpus)"}
    )
    config_name: str = attr.ib(
        default="bert-base-cased",
        metadata={"help": "pretrained model name"}
    )
    batch_size: int = attr.ib(
        default=8,
        metadata={"help": "batch size per process (gpu)"}
    )
    max_steps: int = attr.ib(
        default=0,
        metadata={"help": "maximum number of steps to run training"}
    )
    max_length: int = attr.ib(
        default=512,
        metadata={"help": "maximum token length for input"}
    )
    max_epochs: int = attr.ib(
        default=1,
        metadata={"help": "maximum number of epochs to run training"}
    )
    learning_rate: float = attr.ib(
        default=2e-5,
        metadata={"help": "learning rate"}
    )
    adam_beta1: float = attr.ib(
        default=0.9,
        metadata={"help": "beta_1 parameter for Adam optimizer"}
    )
    adam_beta2: float = attr.ib(
        default=0.999,
        metadata={"help": "beta_2 parameter for Adam optimizer"}
    )
    adam_eps: float = attr.ib(
        default=1e-8,
        metadata={"help": "epsilon parameter for Adam optimizer"}
    )
    weight_decay: float = attr.ib(
        default=0.0,
        metadata={"help": "weight decay for optimizer"}
    )
    dropout: float = attr.ib(
        default=0.0,
        metadata={"help": "dropout rate for regularization"}
    )
    warmup_steps: int = attr.ib(
        default=0,
        metadata={"help": "warmup steps out of total training steps"}
    )
    warmup_ratio: float = attr.ib(
        default=0.0,
        metadata={"help": "warmup ratio out of training steps"}
    )
    validate_ratio: float = attr.ib(
        default=1.0,
        metadata={"help": "validation frequency per epoch"}
    )
    gradient_accumulation_steps: int = attr.ib(
        default=1,
        metadata={"help": "gradient accumulation steps for memory efficiency"}
    )
    checkpoint: str = attr.ib(
        default="",
        metadata={"help": "path to checkpoint"}
    )
    seed: int = attr.ib(
        default=-1,
        metadata={"help": "seed for reproducibility (do not fix if negative)"}
    )
    max_turns: int = attr.ib(
        default=True,
        metadata={"help": "max number of dialog turns to parse"}
    )
    use_special_tokens: bool = attr.ib(
        default=True,
        metadata={"help": "flag for adding special tokens for turn seperator"}
    )
    weights_to_keep: int = attr.ib(
        default=1,
        metadata={"help": "number of weights to keep"}
    )
    flooding: float = attr.ib(
        default=0.0,
        metadata={"help": "flooding parameter for regularization"}
    )

    @staticmethod
    def check_exists(attribute, value):
        if not os.path.isfile(value):
            raise ValueError("{}: {} does not exist ...".format(attribute.name, value))

    @train_file.validator
    def check(self, attribute, value):
        self.check_exists(attribute, value)

    @dev_file.validator
    def check(self, attribute, value):
        self.check_exists(attribute, value)
    
    @devtest_file.validator
    def check(self, attribute, value):
        self.check_exists(attribute, value)

    @warmup_ratio.validator
    def check(self, attribute, value):
        if value > 1.:
            raise ValueError("{} must be less than 1 but received {}.".format(
                attribute.name, value
            ))
    
    @validate_ratio.validator
    def check(self, attribute, value):
        if 0.0 <= value <= 1.0:
            raise ValueError("{} must be in range [0.0, 1.0] but received {}.".format(
                attribute.name, value
            ))

def get_parser():
    """Returns argparse.ArgumentParser with a little hack to 
       change the option string for multi-word arguments.

       ex. --local-rank -> [--local-rank, --local_rank]
       
       This was done to accomodate some arguments passed from
       torch.distributed.
    """
    parser = datargs.make_parser(Options)
    for idx, action in enumerate(parser._actions):
        opt = action.option_strings[0][2:]
        if '-' in opt:
            parser._actions[idx].option_strings = ["--{}".format(opt), "--{}".format(opt.replace('-', '_'))]
            parser._option_string_actions["--{}".format(opt.replace('-', '_'))] = parser._actions[idx]
    return parser

def get_logger(args):
    logging.basicConfig(
        filename="{}/disambiguate_{}_{}.log".format(
            args.log_path,
            args.config_name,
            datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        ), 
        filemode='w', 
        format="%(asctime)s | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s","%Y-%m-%d %H:%M:%S")
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)    
    return logger
