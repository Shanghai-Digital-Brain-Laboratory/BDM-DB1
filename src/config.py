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

"""Basic model functio
"""

from argparse import ArgumentParser


def str2bool(x):
    assert x == "True" or x == "False"
    return True if x == "True" else False


def get_parser_for_basic_args():
    parser = ArgumentParser("Basic Configuration")

    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer_xl"], # Support for GPT2-based model my comming soon
        default="transformer_xl",
        help="Choose the language model to use.",
    )

    parser.add_argument("--load-dir", type=str, help="Path of checkpoint to load.")

    # Text preprocessing
    parser.add_argument("--text-vocab-size", type=int, default=32000)
    parser.add_argument("--pretrained-tokenizer_name", type=str)
    parser.add_argument("--tokenizer-save-path", type=str)
    parser.add_argument("--train-tokenizer", type=str2bool, default=False)

    # Vision Processing
    parser.add_argument("--vision-num-input-channels", type=int, default=3)
    parser.add_argument("--vision-patch-size", type=int, default=16)
    parser.add_argument("--vision-position-vocab-size", type=int, default=128)
    parser.add_argument("--vision-hidden-dropout-prob", type=float, default=0.5)
    parser.add_argument("--eval-ic-iter", type=int, default=0)
    parser.add_argument("--eval-vqa-iter", type=int, default=0)

    # Scalar tokenizer
    parser.add_argument("--num-discrete-values", type=int, default=1024)
    # (ziyu): since in Gato, it is true.
    parser.add_argument("--overlap-with-text", type=str2bool, default=True)

    #   discretization
    parser.add_argument("--num-continuous-bin", type=int, default=1024)
    parser.add_argument("--discretize-mu", type=float, default=100.0)
    parser.add_argument("--discretize-M", type=float, default=256.0)

    # language model
    parser.add_argument(
        "--n-embed",
        type=int,
        default=768,
        help="Vocabulary size of the GPT-2 model. Defines the "
        "number of different tokens that can be represented "
        "by the`inputs_ids` passed when calling",
    )
    parser.add_argument(
        "--n-position",
        type=int,
        default=1024,
        help="The maximum sequence length that this model "
        "might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 "
        "or 2048)",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=12,
        help="Number of hidden layers in the Transformer encoder",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=12,
        help="Number of attention heads for each attention layer in "
        "the Transformer encoder.",
    )
    parser.add_argument(
        "--n-inner",
        type=int,
        default=None,
        help="Dimensionality of the inner feed-forward layers. "
        "`None` will set it to 4 times n_embd",
    )
    parser.add_argument(
        "--activation-fn",
        type=str,
        default="gelu",
        help="Activation function, to be selected in the "
        'list `["relu", "silu", "gelu", "tanh", '
        '"gelu_new"]`.',
    )
    parser.add_argument(
        "--resid-pdrop",
        type=float,
        default=0.1,
        help="The dropout probability for all fully "
        "connected layers in the embeddings, encoder, "
        "and pooler.",
    )
    parser.add_argument(
        "--attn-pdrop",
        type=float,
        default=0.1,
        help="The dropout ratio for the embeddings.",
    )
    parser.add_argument(
        "--embd-pdrop",
        type=float,
        default=0.1,
        help="The dropout ratio for the attention.",
    )

    parser.add_argument(
        "--layer-norm-epsilon",
        type=float,
        default=1e-5,
        help="The epsilon to use in the layer normalization layers.",
    )

    parser.add_argument("--fp16", type=str2bool, default=True)

    # TransformerXL args
    parser.add_argument(
        "--mem-len",
        type=int,
        default=None,
        help="the memory length used during evaluation",
    )
    parser.add_argument(
        "--pre-lnorm", type=str2bool, default=True, help="The position of layernorm"
    )
    parser.add_argument(
        "--same-length",
        type=str2bool,
        default=True,
        help="Whether to use same length in attention masks",
    )
    parser.add_argument(
        "--untie-r",
        type=str2bool,
        default=False,
        help="Whether to use disjoined relative positional vector u, v",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.1,
        help="Dropout of embeddings and ffn in transformer XL",
    )
    parser.add_argument(
        "--dropattn", type=float, default=0.0, help="Dropout of attention output"
    )

    parser.add_argument(
        "--use-deepnorm", type=str2bool, default=False,
        help="Whether to use DeepNorm described in https://arxiv.org/pdf/2203.00555.pdf"
    )

    parser.add_argument(
        "--share-input-output-embedding", type=str2bool, default=False,
        help="Whether to share embedding weights between input and output layer"
    )

    return parser
