import torch

def parse(parser):
    parser.add_argument(
        "--with_image", action="store_true", help="Using MultiModalTransformer"
    )
    parser.add_argument(
        "--task_type", type=str, default="response-generation", help="response-generation"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    # Text Encoder
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="roberta-base",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--add_special_tokens",
        default="/ext/dstc10/yschoi/simmc2_special_tokens.json",
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=-1,
        help="lr < 0, freezing backbone network"
    )
    parser.add_argument(
        "--hidden_dim",
        default=512,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )

    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Optimizer
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2,
        type=int,
        help="Total number of training epochs to perform.",
    )

    # Save 
    parser.add_argument("--save_freq", type=int, default=1000,)
    parser.add_argument("--model_checkpoint_dir", type=str, default="/ext/dstc10/models")

    # Logging
    parser.add_argument("--summary_dir", type=str, default="/ext/dstc10/summary")

    # Cuda
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    args = parser.parse_args()

    return args