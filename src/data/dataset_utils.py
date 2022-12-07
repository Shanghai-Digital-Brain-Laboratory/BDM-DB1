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

""" Most of the code here has been copied from:
    https://github.com/google-research/albert/blob/master/create_pretraining_data.py
    with some modifications. This version is also adapted from Megatron-LM"""

from typing import Tuple

import math
import os
import time
import collections

import numpy as np
import torch

from src.data.blendable_dataset import BlendableDataset
from src.data.indexed_dataset import make_dataset as make_indexed_dataset
from src.data.rl_dataset import RLDataset, RLFullDataset
from src.data.gpt_dataset import GPTDataset
from src.mpu import print_rank_0

DSET_TYPE_BERT = "standard_bert"
DSET_TYPE_ICT = "ict"
DSET_TYPE_T5 = "t5"

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_ICT, DSET_TYPE_T5]

DATASET_CREATORS = {
    "rl": RLDataset,
    "rl_task_suite": RLDataset,
    "nlp": GPTDataset,
}


def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    global_batch_size: int,
    get_build_dataset_fn=None,
    valid_no_blend=False,
) -> Tuple[BlendableDataset, BlendableDataset, BlendableDataset]:
    """Build train, valid, and test datasets, and return a 3-tuple (BlendableDataset)"""

    # Single dataset.
    if len(data_prefix) == 2:
        return _build_train_valid_test_datasets(
            data_prefix[0],
            data_prefix[1],
            data_impl,
            splits_string,
            train_valid_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
            get_build_dataset_fn,
            valid_no_blend=valid_no_blend,
        )

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_type_and_num_samples(
        data_prefix, train_valid_test_num_samples
    )
    (
        prefixes,
        dataset_types,
        weights,
        datasets_train_valid_test_num_samples,
    ) = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        print_rank_0("Build train valid test dataset for {}".format(prefixes[i]))
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i],
            dataset_types[i],
            data_impl,
            splits_string,
            datasets_train_valid_test_num_samples[i],
            seq_length,
            seed,
            skip_warmup,
            get_build_dataset_fn,
        )
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(
            train_datasets, weights, global_batch_size=global_batch_size
        )
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(
            valid_datasets, weights, global_batch_size=global_batch_size
        )

        if valid_no_blend:
            valid_ds_dict = {}
            for i in range(len(prefixes)):
                valid_ds_dict[dataset_types[i]] = valid_datasets[i]
            blending_valid_dataset = (blending_valid_dataset, valid_ds_dict)

    blending_test_dataset = None
    if len(test_datasets) == len(weights):
        blending_test_dataset = BlendableDataset(
            test_datasets, weights, global_batch_size=global_batch_size
        )

    return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    data_prefix,
    dataset_type,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    get_build_dataset_fn=None,
    valid_no_blend=False,
):
    """Build train, valid, and test datasets."""
    if dataset_type in ["nlp", "rl", "rl_task_suite"]:
        if dataset_type == "nlp":
            # Indexed dataset.
            indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

            total_num_of_documents = indexed_dataset.sizes.shape[0]
        elif dataset_type in ["rl", "rl_task_suite"]:
            assert get_build_dataset_fn is not None
            full_dataset = get_build_dataset_fn(dataset_type)(
                env_name=data_prefix, seq_length=seq_length
            )
            total_num_of_documents = len(full_dataset)
            # ugly modification to use code beneath
            get_build_dataset_fn = None
            indexed_dataset = full_dataset

        splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    elif dataset_type in ["ic", "vqa"]:
        # fake splits
        indexed_dataset = None
        splits = [0, 1, 2, 2]
    else:
        raise ValueError("Unknown dataset type {}".format(dataset_type))

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name, dataset_type):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(
                start=splits[index], stop=splits[index + 1], step=1, dtype=np.int64
            )

            if get_build_dataset_fn is None:
                dataset_cls = DATASET_CREATORS[dataset_type]
            else:
                dataset_cls = get_build_dataset_fn(dataset_type)
            dataset = dataset_cls(
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
            )
        return dataset

    train_dataset = build_dataset(0, "train", dataset_type)
    valid_dataset = build_dataset(1, "valid", dataset_type)
    test_dataset = build_dataset(2, "test", dataset_type)

    if valid_no_blend:
        return (
            train_dataset,
            (valid_dataset, {dataset_type: valid_dataset}),
            test_dataset,
        )
    else:
        return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    print_rank_0(
        " > finished creating indexed dataset in {:4f} "
        "seconds".format(time.time() - start_time)
    )
    print_rank_0("    number of documents: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_datasets_weights_and_type_and_num_samples(
    data_prefix, train_valid_test_num_samples
):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 3 == 0
    num_datasets = len(data_prefix) // 3
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    dataset_types = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[3 * i])
        prefixes[i] = (data_prefix[3 * i + 1]).strip()
        dataset_types[i] = (data_prefix[3 * i + 2]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append(
            [
                int(math.ceil(val * weight * 1.005))
                for val in train_valid_test_num_samples
            ]
        )

    return prefixes, dataset_types, weights, datasets_train_valid_test_num_samples


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, "make sure each sample has at least two sentences."

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random
