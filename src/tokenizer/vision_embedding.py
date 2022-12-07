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

"""Implementation of vision encoder with a ResNet-v2 block as described in [Gato][][Gato][https://www.deepmind.com/publications/a-generalist-agent]"""


import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


CLASS_TOKEN_LENGTH = 0


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, patch_size=16, num_channels=3, embed_dim=768, data_type=torch.half
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.data_type = data_type
        patch_size = to_2tuple(patch_size)

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, dtype=data_type
        )
        self.projection = nn.Conv2d(
            64, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=data_type
        )
        self.residual_path = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64, dtype=data_type),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dtype=data_type),
            nn.GroupNorm(num_groups=32, num_channels=64, dtype=data_type),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dtype=data_type),
        )

    def forward(self, pixel_values):
        bsz, c, h, w = pixel_values.shape
        rearranged = einops.rearrange(
            pixel_values,
            pattern="b c (h p1) (w p2) -> (b h w) c p1 p2",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        rearranged = (rearranged - rearranged.mean(dim=(-2, -1), keepdims=True)) / (
            1e-6 + rearranged.std(dim=(-2, -1), keepdims=True)
        )

        rearranged /= math.sqrt(self.patch_size)
        if rearranged.dtype != self.data_type:
            rearranged = rearranged.to(dtype=self.data_type)
        x = self.conv1(rearranged)
        residual = x
        x = self.residual_path(x)
        x = residual + x
        x = self.projection(x)
        x = x.view(bsz, -1, self.embed_dim).contiguous()
        return x


class VisionEmbedding(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config):
        super().__init__()
        data_type = torch.half if config.fp16 else torch.float32
        self.data_type = data_type

        self.patch_embeddings = PatchEmbeddings(
            patch_size=config.vision_patch_size,
            num_channels=config.vision_num_input_channels,
            embed_dim=config.n_embed,
            data_type=data_type,
        )
        self.row_position_embeddings = nn.Embedding(
            config.vision_position_vocab_size, config.n_embed, dtype=data_type
        )
        self.col_position_embeddings = nn.Embedding(
            config.vision_position_vocab_size,
            config.n_embed,
            dtype=data_type,
        )
        self.dropout = nn.Dropout(config.vision_hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values)

        (
            batch_size,
            seq_len,
            _,
        ) = embeddings.size()  # batch_size, seq_len=h*w//patch**2, embed_dim

        # add patch positional encodings

        h0 = height // self.config.vision_patch_size
        w0 = width // self.config.vision_patch_size

        assert seq_len == h0 * w0
        seq_idx = torch.arange(seq_len).to(embeddings.device)
        row_idx = torch.div(seq_idx, w0, rounding_mode="trunc")
        col_idx = seq_idx % w0
        col_idx_high = ((col_idx + 1) / w0 * self.config.vision_position_vocab_size).to(
            torch.int32
        )
        col_idx_low = (col_idx / w0 * self.config.vision_position_vocab_size).to(
            torch.int32
        )
        row_idx_high = ((row_idx + 1) / h0 * self.config.vision_position_vocab_size).to(
            torch.int32
        )
        row_idx_low = (row_idx / h0 * self.config.vision_position_vocab_size).to(
            torch.int32
        )

        if self.training:
            row_pos_encodings, col_pos_encodings = [], []
            for i in range(seq_len):
                col_idx_pick = torch.randint(
                    low=col_idx_low[i],
                    high=col_idx_high[i],
                    size=(batch_size,),
                    device=pixel_values.device,
                )
                col_pos_encodings.append(col_idx_pick.unsqueeze(-1))
                row_idx_pick = torch.randint(
                    low=row_idx_low[i],
                    high=row_idx_high[i],
                    size=(batch_size,),
                    device=pixel_values.device,
                )
                row_pos_encodings.append(row_idx_pick.unsqueeze(-1))

            row_pos_encoding = torch.cat(row_pos_encodings, dim=-1)
            col_pos_encoding = torch.cat(col_pos_encodings, dim=-1)
        else:
            row_pos_encoding = ((row_idx_low + row_idx_high) / 2).int().unsqueeze(0)
            col_pos_encoding = ((col_idx_low + col_idx_high) / 2).int().unsqueeze(0)

        row_pos_encoding = self.row_position_embeddings(row_pos_encoding)
        col_pos_encoding = self.col_position_embeddings(col_pos_encoding)
        embeddings = embeddings + row_pos_encoding + col_pos_encoding

        # embeddings = self.dropout(embeddings)

        return embeddings
