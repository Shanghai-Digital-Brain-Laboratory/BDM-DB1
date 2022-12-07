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

# TODO: reference and description

import torch
import torchvision
import random
import torch.nn.functional as F
import numpy as np

from src.data.input_specs import ICTaskInput, VQATaskInput


class RandomCOCO(torchvision.datasets.CocoCaptions):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform=None,
        target_transform=None,
        transforms=None,
        seq_length: int = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        prompt_items = self.coco.dataset["prompt_items"]
        self.prompt = prompt_items[0]
        self.seq_length = seq_length - len(prompt_items[0])

    def __getitem__(self, index: int):
        assert index < len(self), f"error {index} {len(self)}"
        img, text = super().__getitem__(index)
        text = text[random.randint(0, len(text) - 1)]
        text = torch.IntTensor(text).squeeze()
        if text.shape[-1] >= self.seq_length:
            text = text[..., : self.seq_length]
        else:
            text = F.pad(text, (0, self.seq_length - text.shape[-1]), "constant", 0)

        return {
            "img": img,
            "text": text,
            "prompt": self.prompt,
            "img_id": self.ids[index],
        }


def get_ltor_masks_and_position_ids(
    data,
    eod_token_id,
    full_seq_length,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    seq_length = data.shape[0]
    text_shift = full_seq_length - seq_length
    attention_mask = None

    # Loss mask.
    loss_mask = np.zeros((full_seq_length,), dtype=np.float32)

    loss_mask_text = np.ones(seq_length)
    loss_mask_text[data == eod_token_id] = 0.0
    loss_mask[-seq_length:] = loss_mask_text
    loss_mask[-seq_length - 1] = 1

    # Position ids.
    position_ids = np.zeros((full_seq_length,), dtype=np.int32)
    position_ids[text_shift:] = np.arange(seq_length, dtype=np.int32)

    return attention_mask, loss_mask, position_ids


def get_loss_mask_vqa(
    label,
    eod_token_id,
    eod_mask_loss,
    full_seq_length,
):
    if isinstance(label, list):
        seq_length = len(label)
    else:
        seq_length = label.shape[0]
    loss_mask = np.zeros((full_seq_length,), dtype=np.float32)
    loss_mask_1 = np.ones((seq_length,), dtype=np.float32)
    loss_mask_1[label == eod_token_id] = 0.0

    loss_mask[-seq_length + 1 :] = loss_mask_1[:-1]
    loss_mask[-seq_length] = 1
    return loss_mask


class ICDataset:
    def __init__(self, args, dataset, tokenizer) -> None:
        self.dataset = dataset
        self.args = args
        ICDataset.tokenizer = tokenizer

    def process_text():
        pass

    def __len__(
        self,
    ):
        return len(self.dataset)

    def __getitem__(self, index: int):
        args = self.args
        data = self.dataset[index]

        text = data["text"]
        img = data["img"]
        prompt = data["prompt"]
        img_idx = data["img_id"]

        tokens_ = np.array(text, dtype=np.int32)
        prompt = np.array(prompt, dtype=np.int32)

        tokens = tokens_[:-1]

        _, loss_mask, _ = get_ltor_masks_and_position_ids(
            tokens, ICDataset.tokenizer.eos_token_id, full_seq_length=args.n_position
        )
        labels = np.zeros((args.n_position,), dtype=np.int32)
        labels[(args.n_position - tokens.shape[0]) - 1 :] = tokens_

        img = img.to(dtype=torch.half)
        res = ICTaskInput(
            position_id=None,
            attention_mask=None,
            loss_mask=loss_mask,
            label=labels,
            prompt_seq=prompt,
            img_seq=img,
            text_seq=tokens,
            img_id_seq=img_idx,
        )

        res.apply(lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x)
        res.apply(lambda x: x[None, ...])
        return res


class VQADataset:
    def __init__(self, args, dataset, tokenizer) -> None:
        self.dataset = dataset
        self.args = args
        VQADataset.tokenizer = tokenizer

    def process_text():
        pass

    def __len__(
        self,
    ):
        return len(self.dataset)

    def __getitem__(self, index: int):
        args = self.args
        data = self.dataset[index]

        ques = data["ques"]
        ans = data["ans"]
        img = data["img"]
        ques_idx = data["ques_id"]
        img_idx = data["img_id"]
        prompt = data["prompt"]
        ques_len = data["ques_len"]

        ans_len = len(ans)

        img = img.to(dtype=torch.half)
        tokens = np.concatenate([ques, ans], axis=-1)[:-1]
        labels = np.zeros((args.n_position,), dtype=np.int32)
        labels[-ans_len:] = ans
        loss_mask = get_loss_mask_vqa(
            ans,
            VQADataset.tokenizer.eos_token_id,
            args.eod_mask_loss,
            full_seq_length=args.n_position,
        )
        # print(f"len {ques.shape[0]}")

        res = VQATaskInput(
            position_id=None,
            attention_mask=None,
            loss_mask=loss_mask,
            prompt_seq=prompt,
            img_seq=img,
            text_seq=tokens,
            label=labels,
            img_id_seq=img_idx,
            ques_id_seq=ques_idx,
            ques_len=ques_len,
        )

        res.apply(lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x)
        res.apply(lambda x: x[None, ...])
        return res
