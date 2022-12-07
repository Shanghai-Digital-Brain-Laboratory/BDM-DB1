# Copyright 2022 Digital Brain Laboratory
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for training. 
Adapted from Megatron-LM's configuration schemes 
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py"""

from src.config import get_parser_for_basic_args, str2bool
import deepspeed


def parse_args():
    parser = get_parser_for_basic_args()
    parser = _add_training_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_deepspeed_args(parser)
    parser = _add_finetune_args(parser)

    args = parser.parse_args()

    if args.weight_decay_incr_style == "constant":
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    if args.save_interval == None:
        args.save_interval = args.eval_interval

    return args


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Batch size per model instance (local batch size). "
        "Global batch size is local batch size times data "
        "parallel size times number of micro batches.",
    )
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Training batch size. If set, it should be a "
        "multiple of micro-batch-size times data-parallel-size. "
        "If this value is None, then "
        "use micro-batch-size * data-parallel-size as the "
        "global batch size. This choice will result in 1 for "
        "number of micro-batches.",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Total number of iterations to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )
    group.add_argument(
        "--dataloader-type",
        type=str,
        default=None,
        choices=["single", "cyclic"],
        help="Single pass vs multiple pass data loader",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer function",
    )
    group.add_argument(
        "--num-workers", type=int, default=0, help="Dataloader number of workers."
    )
    group.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate. Depending on decay style "
        "and initial warmup, the learing rate at each "
        "iteration would be different.",
    )
    group.add_argument(
        "--lr-decay-style",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine"],
        help="Learning rate decay function.",
    )
    group.add_argument(
        "--lr-decay-iters",
        type=int,
        default=None,
        help="number of iterations to decay learning rate over,"
        " If None defaults to `--train-iters`",
    )
    group.add_argument(
        "--lr-decay-samples",
        type=int,
        default=None,
        help="number of samples to decay learning rate over,"
        " If None defaults to `--train-samples`",
    )
    group.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=None,
        help="fraction of lr-warmup-(iters/samples) to use " "for warmup (as a float)",
    )
    group.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=0,
        help="number of iterations to linearly warmup " "learning rate over.",
    )
    group.add_argument(
        "--lr-warmup-samples",
        type=int,
        default=0,
        help="number of samples to linearly warmup " "learning rate over.",
    )
    group.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Old lr warmup argument, do not use. Use one of the"
        "--lr-warmup-* arguments above",
    )
    group.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minumum value for learning rate. The scheduler"
        "clip values below this threshold.",
    )
    group.add_argument(
        "--override-opt_param-scheduler",
        action="store_true",
        help="Reset the values of the scheduler (learning rate,"
        "warmup iterations, minimum learning rate, maximum "
        "number of iterations, and decay style from input "
        "arguments and ignore values from checkpoints. Note"
        "that all the above values will be reset.",
    )
    group.add_argument(
        "--use-checkpoint-opt_param-scheduler",
        action="store_true",
        help="Use checkpoint to set the values of the scheduler "
        "(learning rate, warmup iterations, minimum learning "
        "rate, maximum number of iterations, and decay style "
        "from checkpoint and ignore input arguments.",
    )

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title="regularization")
    group.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for hidden state transformer.",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--start-weight-decay",
        type=float,
        help="Initial weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--end-weight-decay",
        type=float,
        help="End of run weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--weight-decay-incr-style",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Weight decay increment function.",
    )
    group.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping based on global L2 norm.",
    )
    group.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="First coefficient for computing running averages "
        "of gradient and its square",
    )
    group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="Second coefficient for computing running averages "
        "of gradient and its square",
    )
    group.add_argument(
        "--adam-eps",
        type=float,
        default=1e-08,
        help="Term added to the denominator to improve" "numerical stability",
    )
    group.add_argument(
        "--sgd-momentum", type=float, default=0.9, help="Momentum factor for sgd"
    )

    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title="validation")
    group.add_argument(
        "--split",
        type=str,
        default="969, 30, 1",
        help="Comma-separated list of proportions for training,"
        " validation, and test split. For example the split "
        "`90,5,5` will use 90%% of data for training, 5%% for "
        "validation and 5%% for test.",
    )
    group.add_argument(
        "--eval-iters",
        type=int,
        default=100,
        help="Number of iterations to run for evaluation" "validation/test for.",
    )
    group.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Interval between running evaluation on " "validation set.",
    )
    group.add_argument(
        "--eval-env-names", nargs="*", default=[], help="RL env names that used to test"
    )

    return parser


def _add_dataset_args(parser):
    group = parser.add_argument_group(title="dataset")
    # XXX(ziyu): now only test mmap
    group.add_argument(
        "--data-path",
        nargs="*",
        default=None,
        help="Path to the training dataset. Accepted format:"
        "1) a single data path with its dataset type, 2) multiple datasets in the"
        "form: dataset1-weight dataset1-path dataset1-type dataset2-weight "
        'dataset2-path dataset2-type ..., dataset types currently are ["rl", "nlp"]',
    )
    group.add_argument(
        "--data-impl",
        type=str,
        default="infer",
        choices=["lazy", "cached", "mmap", "infer"],
        help="Implementation of indexed datasets.",
    )
    # NLP Dataset
    group.add_argument(
        "--reset-position-ids",
        type=str2bool,
        default=False,
        help="Reset posistion ids after end-of-document token.",
    )
    group.add_argument(
        "--reset-attention-mask",
        type=str2bool,
        default=False,
        help="Reset self attention maske after " "end-of-document token.",
    )
    group.add_argument(
        "--eod-mask-loss",
        type=str2bool,
        default=False,
        help="Mask loss for the end of document tokens.",
    )

    # RL Dataset
    group.add_argument("--rl-dataset-cache-dir", type=str, required=True)

    group.add_argument(
        "--use-prompt",
        type=str2bool,
        default=True,
        help="enable prompt or not. (0: off, 1: on)",
    )

    group.add_argument(
        "--prompt-ratio",
        type=float,
        default=0.5,
        help="Ratio of prepending prompt in a rl sequence.",
    )

    group.add_argument(
        "--prompt-prob",
        type=float,
        default=0.25,
        help="Probability of prepending prompt to a rl sequence.",
    )
    group.add_argument(
        "--prompt-at-final-transition-prob",
        type=float,
        default=0.5,
        help="Probability of use the last transitions of an episode.",
    )
    group.add_argument(
        "--mask-prompt-action-loss",
        type=str2bool,
        default=True,
        help="Whether to ignore action loss for prompt actions.",
    )

    group.add_argument(
        "--prompt-strategy",
        type=str,
        default="stochastic_timestep;moving_prompt",
        choices={
            "stochastic_timestep;moving_prompt",
            "stochastic_subseq;moving_prompt",
            "stochastic_timestep;fixed_prompt",
            "stochastic_subseq;fixed_prompt",
        },
    )

    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title="logging")
    group.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Write TensorBoard logs to this directory.",
    )
    group.add_argument(
        "--tensorboard-queue-size",
        type=int,
        default=1000,
        help="Size of the tensorboard queue for pending events "
        "and summaries before one of the ‘add’ calls forces a "
        "flush to disk.",
    )
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title="checkpointing")
    group.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )
    group.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Number of iterations between checkpoint saves.",
    )
    return parser


def _add_deepspeed_args(parser):
    group = parser.add_argument_group(title="_deepspeed_user_args")
    group.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher.",
    )
    group.add_argument(
        "--deepspeed_port", type=int, default=29500, help="Port to initialize deepspeed"
    )

    parser = deepspeed.add_config_arguments(parser)
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title="initialization")
    group.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
    group.add_argument(
        "--init-method-std",
        type=float,
        default=0.02,
        help="Standard deviation of the zero mean normal "
        "distribution used for weight initialization.",
    )
    # parser.add_argument('--init-method-xavier-uniform', action='store_true',
    #                    help='Enable Xavier uniform parameter initialization')
    return parser


def _add_finetune_args(parser):
    group = parser.add_argument_group(title="finetune")
    group.add_argument(
        "--num-rl-fewshot_episodes",
        type=int,
        default=None,
        help="Number of episoes used when finetuning on RL environment",
    )
    return parser