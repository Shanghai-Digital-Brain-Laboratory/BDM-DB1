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

"""Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/vit_dataset.py
"""

import os
import random
import numpy as np
import torch
import torchvision.transforms as T

from src.data.autoaugment import ImageNetPolicy
from src.data.data_samplers import RandomSeedDataset
from PIL import Image, ImageFilter, ImageOps
from src.data.coco_token_dataset import RandomCOCO, ICDataset, VQADataset
from src.data.coco_eval import create_coco_caption_evaluator, create_coco_vqa_evaluator
from src.data.vqa_dataset import CocoVQA


class ClassificationTransform:
    def __init__(self, args, image_size, train=True):
        assert args.fp16 or args.bf16
        self.data_type = torch.half if args.fp16 else torch.bfloat16
        if train:
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(image_size),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    ImageNetPolicy(),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    T.ConvertImageDtype(self.data_type),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    T.ConvertImageDtype(self.data_type),
                ]
            )

    def __call__(self, input):
        output = self.transform(input)
        return output


def get_ic_coco_dataset(
    name,
    data_prefix,
    documents,
    indexed_dataset,
    train_valid_test_num_samples,
    seq_length,
    seed,
    args=None,
    tokenizer=None,
):
    image_size = args.img_h
    vision_seq_length = (args.img_h // args.patch_dim) * (args.img_w // args.patch_dim)
    seq_length = args.n_position - vision_seq_length + 1

    if name == "train":
        train_transform = ClassificationTransform(args, image_size)
        # train_anno_path, train_img_path, train_token_path
        train_token_path = os.path.join(
            data_prefix, "token_data/train_caption_token.json"
        )
        train_img_path = os.path.join(data_prefix, "train2014/")
        train_anno_path = os.path.join(
            data_prefix, "annotations/captions_train2014.json"
        )
        train_img_data = RandomCOCO(
            root=train_img_path,
            annFile=train_token_path,
            transform=train_transform,
            seq_length=seq_length,
        )
        train_data = ICDataset(args, train_img_data, tokenizer)
        train_data = RandomSeedDataset(args, train_data)
        return train_data
    elif name == "valid":
        val_transform = ClassificationTransform(args, image_size, train=False)
        val_token_path = os.path.join(data_prefix, "token_data/val_caption_token.json")
        val_img_path = os.path.join(data_prefix, "val2014/")
        val_anno_path = os.path.join(data_prefix, "annotations/captions_val2014.json")
        create_coco_caption_evaluator(val_anno_path)
        val_img_data = RandomCOCO(
            root=val_img_path,
            annFile=val_token_path,
            transform=val_transform,
            seq_length=seq_length,
        )
        val_data = ICDataset(args, val_img_data, tokenizer)
        return val_data
    else:
        assert name == "test", f"error {name}"
        return None


def get_vqa_v2_dataset(
    name,
    data_prefix,
    documents,
    indexed_dataset,
    train_valid_test_num_samples,
    seq_length,
    seed,
    args=None,
    tokenizer=None,
):
    image_size = args.img_h
    vision_seq_length = (args.img_h // args.patch_dim) * (args.img_w // args.patch_dim)
    seq_length = args.n_position - vision_seq_length + 1

    if name == "train":
        train_transform = ClassificationTransform(args, image_size)
        # train_anno_path, train_img_path, train_token_path
        train_ques_path = os.path.join(
            data_prefix, "token/v2_OpenEnded_mscoco_train2014_questions.json"
        )
        train_answer_path = os.path.join(
            data_prefix, "token/v2_mscoco_train2014_annotations.json"
        )
        train_img_path = os.path.join(data_prefix, "coco-2014")
        train_img_data = CocoVQA(
            root=train_img_path,
            quesFile=train_ques_path,
            annFile=train_answer_path,
            transform=train_transform,
            seq_length=seq_length,
        )
        train_data = VQADataset(args, train_img_data, tokenizer)
        train_data = RandomSeedDataset(args, train_data)
        return train_data
    elif name == "valid":
        val_transform = ClassificationTransform(args, image_size, train=False)
        val_ques_path = os.path.join(
            data_prefix, "token/v2_OpenEnded_mscoco_val2014_questions.json"
        )
        val_answer_path = os.path.join(
            data_prefix, "token/v2_mscoco_val2014_annotations.json"
        )
        val_img_path = os.path.join(data_prefix, "coco-2014")
        val_img_data = CocoVQA(
            root=val_img_path,
            quesFile=val_ques_path,
            annFile=val_answer_path,
            transform=val_transform,
            seq_length=seq_length,
        )
        val_data = VQADataset(args, val_img_data, tokenizer)
        create_coco_vqa_evaluator(val_img_data.vqa)
        return val_data
    else:
        assert name == "test", f"error {name}"
        return None
