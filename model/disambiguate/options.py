import attr


@attr.s
class TrainingOptions:
    debug: bool = attr.ib(
        default=False,
        metadata={"help": "flag for debugging (runs on CPU)"}
    )
    checkpoint: str = attr.ib(
        default=None,
        metadata={"help": "path to last checkpoint"}
    )
    do_tune: bool = attr.ib(
        default=False,
        metadata={"help": "flag for initial learning rate tuning"}
    )
    do_train: bool = attr.ib(
        default=False,
        metadata={"help": "flag for training (train/validate)"}
    )
    do_test: bool = attr.ib(
        default=False,
        metadata={"help": "flag for evaluating (test)"}
    )
    fashion_metadata_path: str = attr.ib(
        default="../../data/fashion_prefab_metadata_all.json",
        metadata={"help": "path to fashion metadata"}
    )
    furniture_metadata_path: str = attr.ib(
        default="../../data/furniture_prefab_metadata_all.json",
        metadata={"help": "path to furniture metadata"}
    )
    tokenizer_path: str = attr.ib(
        default="./data/special_tokens.json",
        metadata={"help": "path to multitask tokenizer special tokens dict"}
    )
    train_raw_path: str = attr.ib(
        default="../../data/simmc2_dials_dstc10_train.json",
        metadata={"help": "path to raw train file"}
    )
    dev_raw_path: str = attr.ib(
        default="../../data/simmc2_dials_dstc10_dev.json",
        metadata={"help": "path to raw dev (validation) file"}
    )
    devtest_raw_path: str = attr.ib(
        default="../../data/simmc2_dials_dstc10_devtest.json",
        metadata={"help": "path to raw devtest file"}
    )
    train_processed_path: str = attr.ib(
        default="./data/processed_baseline_train.json",
        metadata={"help": "path to processed train file"}
    )
    dev_processed_path: str = attr.ib(
        default="./data/processed_baseline_dev.json",
        metadata={"help": "path to processed dev (validation) file"}
    )
    devtest_processed_path: str = attr.ib(
        default="./data/processed_baseline_devtest.json",
        metadata={"help": "path to processed devtest file"}
    )
    target_output_path: str = attr.ib(
        default="./data/targets.txt",
        metadata={"help": "path to targets output"}
    )
    predicted_output_path: str = attr.ib(
        default="./data/predictions.txt",
        metadata={"help": "path to predictions output"}
    )
    log_path: str = attr.ib(
        default="./logs",
        metadata={"help": "path to logs"}
    )
    max_length: int = attr.ib(
        default=1024,
        metadata={"help": "maximum token length"}
    )
    context_length: int = attr.ib(
        default=2, 
        metadata={"help": "length of context"}
    )
    no_multimodal_contexts: bool = attr.ib(
        default=True,
        metadata={
            "help": "flag to use multimodal contexts",
            "dest": "use_multimodal_contexts"
        }
    )
    no_belief_states: bool = attr.ib(
        default=True,
        metadata={
            "help": "flag to use belief states",
            "dest": "use_belief_states"
        }
    )
    no_generate_sys_attr: bool = attr.ib(
        default=True,
        metadata={
            "help": "flag to generate system attribute",
            "dest": "generate_sys_attr"
        }
    )
    config_name: str = attr.ib(
        default="facebook/bart-base",
        metadata={"help": "model name from HF hub"}
    )
    batch_size: int = attr.ib(
        default=4,
        metadata={"help": "batch size per device"}
    )
    max_epochs: int = attr.ib(
        default=2, 
        metadata={"help": "max epochs to train"}
    )
    learning_rate: float = attr.ib(
        default=5e-5,
        metadata={"help": "initial learning rate for optimizer"}
    )
    beta_1: float = attr.ib(
        default=0.9,
        metadata={"help": "first moment parameter in Adam"}
    )
    beta_2: float = attr.ib(
        default=0.999,
        metadata={"help": "second moment parameter in Adam"}
    )
    weight_decay: float = attr.ib(
        default=0.0,
        metadata={"help": "weight decay for regularization"}
    )
    warmup_ratio: float = attr.ib(
        default=0.0,
        validator=lambda i, a, x: 0. <= x and x <= 1.,
        metadata={"help": "linear scheduler warmup ratio"}
    )
    num_workers: int = attr.ib(
        default=0,
        metadata={"help": "number of workers in dataloader"}
    )
    ddp: bool = attr.ib(
        default=False,
        metadata={"help": "distributed data parallel training"}
    )
    fp16: bool = attr.ib(
        default=False,
        metadata={"help": "flag for mixed precision training"}
    )
    seed: int = attr.ib(
        default=42,
        metadata={"help": "seed to fix (do not fix if negative)"}
    )
    deterministic: bool = attr.ib(
        default=True,
        metadata={"help": "flag for determinism"}
    )
    accumulate_grad_batches: int = attr.ib(
        default=1,
        metadata={"help": "gradient accumulation steps for memory efficiency"}
    )
    gpus: str = attr.ib(
        default="0",
        metadata={"help": "gpus to use (refer to pytorch-lightning docs)"}
    )
    log_interval: int = attr.ib(
        default=100,
        metadata={"help": "logging interval"}
    )
    val_check_interval: float = attr.ib(
        default=1.0,
        metadata={"help": "validation check interval for every epoch"}
    )
    top_k: int = attr.ib(
        default=None,
        metadata={"help": "top-k for generation (not recommended)"}
    )
    top_p: float = attr.ib(
        default=0.9,
        metadata={"help": "top-p for generation (recommended)"}
    )
    repetition_penalty: float = attr.ib(
        default=1.0,
        metadata={"help": "repetition penalty for generation"}
    )
    num_return_sequences: int = attr.ib(
        default=1,
        metadata={"help": "number of sequences to generate"}
    )
    mix_lambda: float = attr.ib(
        default=0.01,
        metadata={"help": "loss weight parameter"}
    )