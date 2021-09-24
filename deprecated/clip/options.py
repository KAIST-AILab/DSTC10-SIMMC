import attr


@attr.s
class TrainingOptions:
    debug: bool = attr.ib(
        default=False,
        metadata={"help": "debugging flag (runs on cpu)"}
    )
    checkpoint: str = attr.ib(
        default=None,
        metadata={"help": "path to last checkpoint"}
    )
    do_train: bool = attr.ib(
        default=False,
        metadata={"help": "flag for training (train/validate)"}
    )
    do_test: bool = attr.ib(
        default=False,
        metadata={"help": "flag for evaluating (test)"}
    )
    metadata_path: str = attr.ib(
        default="./data",
        metadata={"help": "directory with metadata"}
    )
    item2meta_dump_path: str = attr.ib(
        default="./clip/data/item2meta.json",
        metadata={"help": "save path for mapping from item to metadata in json format"}
    )
    glove_path: str = attr.ib(
        default="./clip/data/glove.pt",
        metadata={"help": "path to preprocessed pretrained GloVE embeddings"}
    )
    image_path: str = attr.ib(
        default="./clip/logs/scatter.png",
        metadata={"help": "path to save TSNE scatter plot for testing"}
    )
    log_path: str = attr.ib(
        default="./clip/logs",
        metadata={"help": "path to logs"}
    )
    glove_dim: int = attr.ib(
        default=200,
        metadata={"help": "glove embedding dimension"}
    )
    attention_dim: int = attr.ib(
        default=128, 
        metadata={"help": "attention pooler dimension"}
    )
    embedding_dim: int = attr.ib(
        default=768,
        metadata={"help": "embedding dimension"}
    )
    batch_size: int = attr.ib(
        default=128,
        metadata={"help": "batch size per device"}
    )
    max_epochs: int = attr.ib(
        default=800, 
        metadata={"help": "max epochs to train"}
    )
    learning_rate: float = attr.ib(
        default=1e-4,
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
    pool_type: str = attr.ib(
        default='cls',
        metadata={
            "help": "pooling type for attention pooler",
            "choices": ['cls', 'mean']
        }
    )
    separate_domain: bool = attr.ib(
        default=False,
        metadata={"help": "flag to separate domain"}
    )