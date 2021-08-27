import logging 

from typing import Any
from pathlib import Path
from argparse import ArgumentParser, Namespace

import datargs

from pytorch_lightning.utilities.distributed import rank_zero_only

def get_parser(options: Any) -> ArgumentParser:
    ''''
        Returns argparse.ArgumentParser with a little hack to 
        change the option string for multi-word arguments.

        e.g. --local-rank -> [--local-rank, --local_rank]

        This was done to accommodate some arguments passed from 
        torch.distributed.

        Args:
            options <Any>: an attrs class having arguments
        
        Returns:
            parser <ArgumentParser>: an argument parser
    '''
    parser = datargs.make_parser(options)
    for idx, action in enumerate(parser._actions):
        opt = action.option_strings[0][2:]
        if "-" in opt:
            parser._actions[idx].option_strings = [
                "--{}".format(opt),
                "--{}".format(opt.replace("-", "_")),
            ]
            parser._option_string_actions[
                "--{}".format(opt.replace("-", "_"))
            ] = parser._actions[idx]
    return parser


def get_args_string(args: Namespace) -> str:
    '''
        Returns parsed argument string to a nice format.

        Args:
            args <Namespace>: parsed arguments

        Returns:
            result <str>: formatted argument string
    '''
    max_str_len = max([len(k)+len(str(v)) for k, v in vars(args).items()])
    width = max_str_len // 2

    top_str = "\n{} Arguments {}\n".format('=' * width, '=' * width)
    result = top_str
    for key, val in sorted(vars(args).items()):
        result += "{key: <{width}} : {val}\n".format(key=key, val=val, width=width)
    result += "{}\n".format("=" * len(top_str))
    return result

@rank_zero_only
def get_logger(log_path: str, exp_name: str, timestamp: str) -> logging.Logger:
    '''
        Returns a logger that prints to stdout as well.

        Args:
            log_path <str>: logger path
            exp_name <str>: experiment name
            timestamp <str>: time stamp -- will be given
        
        Returns:
            logger <logging.Logger>: logger
    '''
    log_dir = Path(log_path)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename="{}/{}_{}.log".format(
            log_path, timestamp, exp_name
        ),
        filemode="w",
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger
