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

"""Implementation of training procedure"""

from typing import Dict, Generator, Type, Tuple, Any
from argparse import Namespace

import torch

from src.checkpointing import save_checkpoint
from src.mpu import print_rank_0
from src.evaluation.evaluate_rl import evalute_one_episode
from src.evaluation.evaluate_vqa import evaluate_vqa
from src.evaluation.evaluate_ic import evaluate_ic
import numpy as np

from src.data.text_decoder import Decoder
from src.tokenizer.text_tokenizer import build_text_tokenizer


def train(
    args: Namespace,
    model: torch.nn.Module,
    train_data_iterator: Generator,
    valid_data_iterator: Tuple[Generator, Dict],
    get_batch_fn: Type,
    sm_writer: Any = None,
):
    if Decoder.__dict__.get("tokenizer") is None or Decoder.tokenizer is None:
        Decoder.tokenizer = build_text_tokenizer(
            args.tokenizer_save_path, False, None, None, None,
        )
    text_decoder = Decoder(args)

    # Iterations.
    iteration = args.iteration
    # print("***** start train")
    while iteration < args.train_iters:
        losses = train_step(args, model, train_data_iterator, get_batch_fn)
        if sm_writer:
            loss = sum(losses) / len(losses)

            sm_writer.add_scalar("Train loss", loss.item(), iteration)

        args.iteration = iteration
        if (
            args.eval_interval
            and iteration % args.eval_interval == 0
            or iteration == args.train_iters - 1
        ):
            evaluate_and_print_results(
                args,
                model,
                valid_data_iterator,
                get_batch_fn,
                iteration,
                sm_writer,
                text_decoder=text_decoder,
            )

        iteration += 1

        if args.save_dir and iteration % args.save_interval == 0:
            save_checkpoint(args, iteration, model)


def train_step(args, model, data_iterator, get_batch_fn):
    model.train()
    losses, _, _ = forward_and_backward_step(
        args, model, data_iterator, get_batch_fn, do_backward=True
    )
    return losses


def evaluate_and_print_results(
    args: Namespace,
    model: torch.nn.Module,
    data_iterator: Tuple[Generator, Dict],
    get_batch_fn: Type,
    iteration_log: int,
    sm_writer=None,
    text_decoder=None,
):
    data_iterator, data_iters_dict = data_iterator

    model.eval()

    with torch.no_grad():
        iteration = 0
        total_loss = 0
        sub_loss = {}
        episode_return = {k: [] for k in args.eval_env_names}
        episode_length = {k: [] for k in args.eval_env_names}

        while iteration < args.eval_iters:
            iteration += 1

            # get total valid loss
            forward_ret = forward_and_backward_step(
                args,
                model,
                data_iterator,
                get_batch_fn,
                do_backward=False,
                return_all=True,
            )
            loss_list, logits_list, input_data_list = forward_ret

            loss_list = [l.cpu().item() for l in loss_list]

            total_loss = total_loss + np.mean(loss_list)

            if torch.distributed.get_rank() == 0:

                # eval rl tasks
                for env_name in args.eval_env_names:
                    ep_ret, ep_len = evalute_one_episode(
                        args, model, env_name
                    )
                    episode_return[env_name].append(ep_ret)
                    episode_length[env_name].append(ep_len)

        # XXX: be careful with this when use parallel
        total_loss /= args.eval_iters

    print_rank_0("Validation loss: {:.6E}".format(total_loss))
    print_rank_0(f"Validation loss all: {sub_loss}")

    # eval ic
    if (
        args.eval_ic_iter > 0
        and torch.distributed.get_rank() == 0
        and data_iters_dict.get("ic")
    ):
        eval_ic_result = evaluate_ic(
            args,
            model,
            data_iters_dict["ic"],
            text_decoder,
            get_batch_fn,
            skip_metrics=["SPICE"],
            eval_iter=args.eval_ic_iter,
            print_first_k=10,
        )

    # eval vqa
    if (
        args.eval_vqa_iter > 0
        and torch.distributed.get_rank() == 0
        and data_iters_dict.get("vqa") is not None
    ):
        eval_vqa_result = evaluate_vqa(
            args,
            model,
            data_iters_dict["vqa"],
            text_decoder,
            get_batch_fn,
            eval_iter=args.eval_vqa_iter,
            print_first_k=10,
        )

    if sm_writer:
        sm_writer.add_scalar("validation/loss", total_loss, iteration_log)
        for k, l in sub_loss.items():
            sm_writer.add_scalar(
                f"validation/{k}_loss", l / args.eval_iters, iteration_log
            )

        # ic info
        if args.eval_ic_iter > 0 and data_iters_dict.get("ic") is not None:
            for method, val in eval_ic_result.items():
                sm_writer.add_scalar(f"validation ic/{method}", val, iteration_log)

        # vqa info
        if args.eval_vqa_iter > 0 and data_iters_dict.get("vqa") is not None:
            sm_writer.add_scalar(
                f"validation vqa/overall", eval_vqa_result["overall"], iteration_log
            )

            for method, val in eval_vqa_result["perAnswerType"].items():
                sm_writer.add_scalar(f"validation vqa/{method}", val, iteration_log)

        for env_name in args.eval_env_names:
            ep_ret = np.mean(episode_return[env_name])
            ep_len = np.mean(episode_length[env_name])
            print("Env: {}, EpRet: {:.4f}, EpLen: {}.".format(env_name, ep_ret, ep_len))
            sm_writer.add_scalar(
                "validation rl/{}/episode_return".format(env_name),
                ep_ret,
                iteration_log,
            )
            sm_writer.add_scalar(
                "validation rl/{}/episode_length".format(env_name),
                ep_len,
                iteration_log,
            )


def forward_and_backward_step(
    args, model, data_iterator, get_batch_fn, do_backward=True, return_all=False
):
    loss_list = []
    logits_list = []
    input_data_list = []
    ga_steps = model.gradient_accumulation_steps()
    for i in range(ga_steps):
        input_data = get_batch_fn(args, data_iterator)

        # img_id_seq = input_data[0].img_id_seq
        # img_seq = input_data[0].img_seq
        # print("==== save image....", img_seq.shape[0])
        # for j in range(img_seq.shape[0]):
        #     with open(f"./img/{img_id_seq[j].item()}.png", "wb") as f:
        #         print(f"save img at {img_id_seq[j].item()}.png", img_seq[j].dtype, input_data[0].ques_id_seq[j])
        #         vutils.save_image(img_seq[j].to(device="cpu", dtype=torch.float32), f)

        logits, loss = model(input_data)

        if do_backward:
            model.backward(loss)
            model.step()

        if return_all:
            input_data_list.append(input_data)
            logits_list.append(logits)
        loss_list.append(loss)

    # WARNING: Never use different number of returns !
    if return_all:
        return loss_list, logits_list, input_data_list
    else:
        return loss_list, None, None
