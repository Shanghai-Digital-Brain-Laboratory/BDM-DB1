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

"""The pretraining code."""


from functools import partial
from typing import List, Tuple, Generator, Dict
import numpy as np
from src.data.blendable_dataset import BlendableDataset
from src.data.input_specs import GatoInputBase
from src.data.data_samplers import (
    build_pretraining_data_loader,
    build_naive_data_loader,
)
from src.data.dataset_utils import build_train_valid_test_datasets
from src.data.vit_dataset import get_ic_coco_dataset, get_vqa_v2_dataset
import torch
import torch.nn as nn
from src.train_utils.optimizer_param_scheduler import OptimizerParamScheduler

from src.tokenizer.text_tokenizer import build_text_tokenizer
from src.train_utils.train import train
from src.train_utils.train_config import parse_args
from torch.optim import Adam, SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from src import mpu
from src.mpu import print_rank_0, print_with_rank
import deepspeed
from src.data.rl_dataset import RLFullDataset, RLTaskSuiteDataset
from src.data.gpt_dataset import GPTDataset
from src.tokenizer.scalar_tokenizer import ContinuousScalarTokenizer

from src.model.transformer_xl import PositionalEmbedding
from src.model import TransformerXL


def get_model(args):
    if args.model == "gpt":
        raise NotImplementedError
    elif args.model == "transformer_xl":
        model = TransformerXL(args)
    else:
        raise NotImplementedError

    return model


def get_param_groups(args, model):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d)
    blacklist_weight_modules = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.Embedding,
        PositionalEmbedding,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # Special case for Transformer XL share embedding
    decay = param_dict.keys() & decay
    no_decay = param_dict.keys() & no_decay
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() ^ union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )
    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [
                param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    return optim_groups


def get_optimizer(args, model):
    # Base optimizer.
    param_groups = get_param_groups(args, model)

    if args.optimizer == "adam":
        optimizer = Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
        )
    else:
        raise Exception("{} optimizer is not supported.".format(args.optimizer))

    return optimizer


def get_optimizer_param_scheduler(args, optimizer):
    """Build the learning rate scheduler."""

    # Iteration-based training.
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
    )

    return opt_param_scheduler


def get_tensorboard_writer(args):
    return SummaryWriter(
        log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
    )


def get_data_iterators(
    args,
) -> Tuple[Generator, Tuple[Generator, Dict], BlendableDataset]:
    """Create iterators for training and validate datsets, also a dict of datasets for evaluation (one for each task)"""

    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x

    if args.iteration > 0 and args.consumed_train_samples == 0:
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        args.consumed_valid_samples = (
            (args.iteration // args.eval_interval)
            * args.eval_iters
            * args.global_batch_size
        )

    train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    ]
    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

    tokenizer, cont_tokenizer = get_tokenizers(args)

    def get_build_dataset_fn(dataset_type):
        if dataset_type == "rl":
            return partial(
                RLFullDataset,
                tokenizers=(tokenizer, cont_tokenizer),
                overlap_with_text=args.overlap_with_text,
                num_discrete_values=args.num_discrete_values,
                cache_path=args.rl_dataset_cache_dir,
                prompt_ratio=args.prompt_ratio,
                prompt_prob=args.prompt_prob,
                prompt_at_final_transition_prob=args.prompt_at_final_transition_prob,
                mask_prompt_action_loss=args.mask_prompt_action_loss,
                use_prompt=args.use_prompt,
                prompt_strategy=args.prompt_strategy.split(";")[0],
                vision_patch_size=args.vision_patch_size,
            )
        elif dataset_type == "rl_task_suite":
            build_rl_full_dataset_fn = get_build_dataset_fn("rl")
            return partial(
                RLTaskSuiteDataset, build_rl_full_dataset_fn=build_rl_full_dataset_fn
            )
        elif dataset_type == "nlp":
            return partial(
                GPTDataset,
                eos_token_id=tokenizer.eos_token_id,
                reset_position_ids=args.reset_position_ids,
                reset_attention_mask=args.reset_attention_mask,
                eod_mask_loss=args.eod_mask_loss,
            )
        elif dataset_type == "ic":
            return partial(get_ic_coco_dataset, args=args, tokenizer=tokenizer)
        elif dataset_type == "vqa":
            return partial(get_vqa_v2_dataset, args=args, tokenizer=tokenizer)
        else:
            raise ValueError("Unknown Dataset Type: {}.".format(dataset_type))

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        args.data_path,
        args.data_impl,
        args.split,
        train_val_test_num_samples,
        seq_length=args.n_position,
        seed=args.seed,
        global_batch_size=args.global_batch_size,
        skip_warmup=True,
        get_build_dataset_fn=get_build_dataset_fn,
        valid_no_blend=True,
    )
    print_rank_0("Size of Training set: {}, total_train_samples: {}".format(
        len(train_ds), train_samples))
    valid_ds, valid_ds_dict = valid_ds

    train_dataloader = build_pretraining_data_loader(
        args, train_ds, args.consumed_train_samples, train_samples
    )
    valid_dataloader = build_pretraining_data_loader(
        args, valid_ds, args.consumed_valid_samples, train_samples, eval=True
    )
    valid_dl_dict = {}
    for k, d in valid_ds_dict.items():
        valid_dl_dict[k] = build_naive_data_loader(d)

    # test_dataloader = build_pretraining_data_loader(args, test_ds, 0)
    dl_type = args.dataloader_type

    if train_dataloader is not None:
        train_data_iterator = (
            iter(train_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(train_dataloader))
        )
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = (
            iter(valid_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(valid_dataloader))
        )
        valid_data_iterator = (valid_data_iterator, valid_dl_dict)
    else:
        valid_data_iterator = None

    # if test_dataloader is not None:
    #     test_data_iterator = (
    #         iter(test_dataloader)
    #         if dl_type == "single"
    #         else iter(cyclic_iter(test_dataloader))
    #     )
    # else:
    #     test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_ds  # test_data_iterator


def get_tokenizers(args):
    tokenizer = build_text_tokenizer(
        args.tokenizer_save_path,
        False,
        None,
        None,
        None,
    )
    # assert tokenizer.vocab_size == args.text_vocab_size
    cont_tokenizer = ContinuousScalarTokenizer(
        args.num_continuous_bin, args.discretize_mu, args.discretize_M
    )
    return tokenizer, cont_tokenizer


def get_batch(args, data_iterator):
    """Generate a batch"""
    data: List[GatoInputBase] = next(data_iterator)
    for _x in data:
        _x.to(device=args.device)
    if args.fp16:
        for _x in data:
            _x.apply(lambda x: x.to(dtype=torch.half) if x.dtype == torch.float else x)
    return data


def main(args):
    model = get_model(args)

    optimizer = get_optimizer(args, model)
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)

    print("deep args", args.deepspeed_port)

    deepspeed.init_distributed(distributed_port=args.deepspeed_port)

    mpu.initialize_model_parallel()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()]),
            ),
            flush=True,
        )
    print_rank_0(" ============= MPU_INIT ==============")
    model_engine, _, _, _ = deepspeed.initialize(
        args,
        model,
        model_parameters=model.parameters(),
        mpu=mpu,
        optimizer=optimizer,
        lr_scheduler=opt_param_scheduler,
    )
    print_rank_0(" ============= DS_INIT ==============")

    assert args.fp16 == model_engine.fp16_enabled()
    if args.load_dir:
        load_path, client_state = model_engine.load_checkpoint(
            args.load_dir,
            tag="1.2B_240k"
            # load_optimizer_states=False,
            # load_lr_scheduler_states=False
        )
        args.iteration = client_state["iteration"]
        # args.iteration = 0
        print_with_rank("load at iter: {}, lr: {}".format(client_state["iteration"], model_engine.client_lr_scheduler.get_lr()))
    args.device = model_engine.device
    from torch.distributed import get_rank

    (
        train_data_iterator,
        valid_data_iterator,
        eval_blendable_dataset,
    ) = get_data_iterators(args)

    if get_rank() == 0:
        tensorboard_writer = get_tensorboard_writer(args)
    else:
        tensorboard_writer = None
    train(
        args,
        model_engine,
        train_data_iterator,
        valid_data_iterator,
        get_batch,
        tensorboard_writer,
    )


if __name__ == "__main__":
    args = parse_args()

    args.iteration = 0
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    args.bf16 = False
    args.fp16 = True

    main(args)
