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

"""Implementation of Text tokenizer and its training code"""

from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer


def build_text_tokenizer(
    save_path: str,
    train_from_scratch: bool = False,
    pretrained_tokenizer_name: Optional[str] = None,
    training_corpus: Optional = None,
    vocab_size: int = 32000,
):
    """
    This is our TextTokenizer which can just load a pretrained tokenizer exists
    in huggingface or train from the old one use corpus_file.
    If there is a directory in save path then it will load it from save path
    and return.

    :param save_path: the directory path to save or load the tokenizer
    :param train_from_scratch: whether the tokenizer will train from new corpus
    :param pretrained_tokenizer_name: the name of the pretrained tokenizer in huggingface to load
    :param training_corpus: a generator of the new corpus data
    :param vocab_size:
    """

    if not train_from_scratch:
        if Path(save_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(save_path)
        else:
            assert pretrained_tokenizer_name is not None
            tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
            tokenizer.save_pretrained(save_path)
    else:
        old_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)

        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        tokenizer.save_pretrained(save_path)

    return tokenizer
