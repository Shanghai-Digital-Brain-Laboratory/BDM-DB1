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

"""Code adapted from 
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/blendable_dataset.py

Implementations of building blendableDataset
"""
 

import time
from typing import List, Optional

import numpy as np
import torch

from src.mpu import print_rank_0, print_with_rank


class BlendableDataset(torch.utils.data.Dataset):
    """A naive implementation of collection of multiple datasets sampled in a weighted round-robin manner."""
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        weights,
        global_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.size = 0
        weights = torch.tensor(weights)
        assert (weights > 0).all()
        weights /= weights.sum()

        if global_batch_size is None:
            global_batch_size = len(datasets)
        else:
            assert global_batch_size >= len(datasets)
        self.sample_batch_size = global_batch_size
        num_samples_one_batch = (global_batch_size * weights).round()
        offset_in_batch = num_samples_one_batch.cumsum(0).int().numpy()
        self.offset_in_batch = np.zeros_like(offset_in_batch)
        self.offset_in_batch[1:] = offset_in_batch[:-1]
        for i, dataset in enumerate(self.datasets):
            self.size += len(dataset)
        print_rank_0("Num of each dataset in a batch: {}".format(num_samples_one_batch))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inner_batch_idx = idx % self.sample_batch_size
        dataset_idx = np.argwhere(self.offset_in_batch <= inner_batch_idx).max()

        # inner_dataset_offset = int(idx - self.dataset_offset[dataset_idx])
        inner_dataset_offset = np.random.randint(
            low=0, high=len(self.datasets[dataset_idx])
        )
        dataset_idx, inner_dataset_offset = int(dataset_idx), int(inner_dataset_offset)
        # print_with_rank(
        #     "idx: {}. ds_idx: {}. inner_ds_idx: {}".format(idx, dataset_idx, inner_dataset_offset))
        return self.datasets[dataset_idx][inner_dataset_offset]

# # Original implementation of Megatron-LM in which an index will be precomputed to determine which inner-dataset should be sampled.
# class BlendableDataset(torch.utils.data.Dataset):
#     def __init__(self, datasets, weights):

#         self.datasets = datasets
#         num_datasets = len(datasets)
#         assert num_datasets == len(weights), f"{len(datasets)}, {weights}"

#         self.size = 0
#         for dataset in self.datasets:
#             self.size += len(dataset)

#         # Normalize weights.
#         weights = np.array(weights, dtype=np.float64)
#         sum_weights = np.sum(weights)
#         assert sum_weights > 0.0
#         weights /= sum_weights

#         # Build indecies.
#         start_time = time.time()
#         assert num_datasets < 255
#         self.dataset_index = np.zeros(self.size, dtype=np.uint8)
#         self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

#         from . import helpers

#         helpers.build_blending_indices(
#             self.dataset_index,
#             self.dataset_sample_index,
#             weights,
#             num_datasets,
#             self.size,
#             torch.distributed.get_rank() == 0
#             if torch.distributed.is_initialized()
#             else True,
#         )
#         print_rank_0(
#             "> elapsed time for building blendable dataset indices: "
#             "{:.2f} (sec)".format(time.time() - start_time)
#         )

#     def __len__(self):
#         return self.size


#     def __getitem__(self, idx):
#         dataset_idx = self.dataset_index[idx]
#         sample_idx = self.dataset_sample_index[idx] % len(self.datasets[dataset_idx])
#         return self.datasets[dataset_idx][sample_idx]
