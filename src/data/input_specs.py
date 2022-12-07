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

"""Input types for DB1 of different modalities and tasks"""
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Optional, Union, List
import numpy as np
import torch
import sys


@dataclass
class GatoInputBase:
    position_id: Optional[torch.tensor]
    attention_mask: Optional[torch.tensor]
    loss_mask: Optional[torch.tensor]
    label: Optional[torch.tensor]

    def get_datasize(self):
        gb = 0
        for e in [self.position_id, self.attention_mask, self.loss_mask, self.label]:
            if e is not None:
                gb += e.element_size() * e.nelement()
        gb /= 1024 * 1024 * 1024
        return gb

    def to(self, **kwargs):
        for k, v in asdict(self).items():
            if v is not None:
                setattr(self, k, v.to(**kwargs))

    def apply(self, fn, *args, **kwargs):
        for k, v in asdict(self).items():
            if v is not None:
                setattr(self, k, fn(v, *args, **kwargs))

    def append(self, other):
        assert type(self).__name__ == type(other).__name__
        for k, v in asdict(self).items():
            if v is None:
                continue
            other_v = getattr(other, k)
            # assert v.ndim == other_v.ndim == 2, (k, v.shape, other_v.shape)
            new_v = torch.cat([v, other_v], dim=0)
            # new_v = np.concatenate([v, other_v], axis=0)
            setattr(self, k, new_v)

    @staticmethod
    def merge_into_one(data2merge: List["GatoInputBase"]):
        t = data2merge[0]
        t.apply(lambda x: [x])
        for i in range(1, len(data2merge)):
            for k, v in asdict(data2merge[i]).items():
                if v is not None:
                    t_v = getattr(t, k)
                    t_v.append(v)
            # set(t, k, t_v)
        return t


@dataclass
class RLTaskInput(GatoInputBase):
    text_seq: Union[List, torch.tensor]
    vision_seq: Union[List, torch.tensor]
    tensor_seq: Union[List, torch.tensor]


@dataclass
class NLPTaskInput(GatoInputBase):
    text_seq: Union[List, torch.tensor]
    text_len: Union[List, torch.tensor]


@dataclass
class ICTaskInput(GatoInputBase):
    """
    propmt format:
        Caption the image: [image] [pic]
    """

    prompt_seq: Union[List, torch.tensor]
    img_seq: Union[List, torch.tensor]
    text_seq: Union[List, torch.tensor]
    img_id_seq: Union[List, torch.tensor]


@dataclass
class VQATaskInput(GatoInputBase):
    """
    prompt format:
        Answer a question after the image: [image]
        Question: [text]
        Answer: [label]
    """

    prompt_seq: Union[List, torch.tensor]
    img_seq: Union[List, torch.tensor]
    text_seq: Union[List, torch.tensor]
    img_id_seq: Union[List, torch.tensor]
    ques_id_seq: Union[List, torch.tensor]
    ques_len: Union[List, torch.tensor]
