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

"""Code for evaluating DB1, currently we only ensure TransformerXL-based model running correctly."""

import copy
from argparse import Namespace
import functools
from typing import List, Tuple, Dict, Union, Optional

import deepspeed
import numpy as np
from src.data.input_specs import RLTaskInput
from src.model import TransformerXL
from src.config import get_parser_for_basic_args, str2bool
from src import mpu
from src.mpu import print_rank_0
from src.train_utils.train_config import _add_dataset_args, _add_deepspeed_args
from src.evaluation.rl.rl_eval_config import _add_rl_eval_args
import gym
import d4rl
import torch
import torch.distributed as dist
from src.tokenizer.scalar_tokenizer import ContinuousScalarTokenizer
from src.data.rl_dataset import (
    RLDataset,
    RLFullDataset,
    _get_action_flag_and_position_id,
)
from src.evaluation.rl.wrapper import LMPromptEnv

import tree
from src.tokenizer.text_tokenizer import build_text_tokenizer
import importlib


dataset = None


def judge_discrete_space(s):
    if isinstance(s, gym.spaces.Discrete):
        return True
    elif isinstance(s, gym.spaces.Box):
        return False
    else:
        raise NotImplementedError


def get_args():
    parser = get_parser_for_basic_args()
    parser = _add_deepspeed_args(parser)
    parser = _add_dataset_args(parser)

    # used for scripts that calling this __file__
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--ckpt-tag", type=str, default=None
    )
    parser.add_argument(
        "--env-name",
        type=str,
        nargs="*",
        help="Environment name to test, which will be pass to gym.make",
    )
    parser.add_argument("--task-suite-name", nargs="*")

    parser = _add_rl_eval_args(parser)
    return parser.parse_args()

def get_model(args):
    if args.model == "gpt":
        raise NotImplementedError("Comming Soon")
    elif args.model == "transformer_xl":
        model = TransformerXL(args)
    else:
        raise NotImplementedError

    return model


def masked_logits_for_action(
    args,
    logits,
    discrete_action: bool,
    action_space,
    env_action_mask: np.ndarray = None,
):
    """MASK LOGITS TO MAKE IT PREDICT ACTION TOKEN"""
    B, L, N = logits.shape
    if not discrete_action:
        if args.overlap_with_text:
            logits[..., : args.text_vocab_size] -= 1e10
        else:
            logits[..., : args.text_vocab_size + args.num_discrete_values] -= 1e10
        logits[..., -1] -= 1e10  # this is the spliter token
    else:
        if args.overlap_with_text:
            # logits[..., args.num_discrete_values :] -= 1e10
            logits[..., action_space.n :] -= 1e10
        else:
            logits[..., : args.text_vocab_size] -= 1e10
            logits[..., args.text_vocab_size + action_space.n :] -= 1e10
        # TODO(ming): mask with real action mask
        if env_action_mask is not None:
            env_action_mask = np.abs(env_action_mask - 1) * 1e10
            logits[:, -1, : action_space.n] = logits[
                :, -1, : action_space.n
            ] - torch.from_numpy(env_action_mask.reshape(1, -1)).to(logits.device)
    return logits


def recover_model_predict_token_to_tokenizer_raw(args, preds, discrete_action: bool):
    if args.overlap_with_text:
        if not discrete_action:
            assert (preds >= args.text_vocab_size).all(), preds
            preds -= args.text_vocab_size - args.num_discrete_values
        else:
            assert (preds < args.num_discrete_values).all()
    else:
        preds -= args.text_vocab_size
    if not discrete_action:
        preds -= args.num_discrete_values
    return preds


def truncate_sequence_by_stepsize(
    current_seq, vision_seq, obs_length, act_length, max_length=None
):
    current_length = len(current_seq)
    stepsize = obs_length + act_length + 1
    return current_seq[stepsize:], vision_seq[1:] if vision_seq is not None else None


def truncate_memory(mems, obs_len, act_len):
    step_size = obs_len + act_len + 1
    res_mems = []
    for mem in mems:
        res_mems.append(mem[:, step_size:])
    return res_mems


def get_action(
    args,
    model,
    current_seq,
    vision_seq,
    cont_tokenizer,
    len_fixed_prompt,
    len_fixed_prompt_img,
    obs_length,
    action_length,
    discrete_action: bool,
    action_space,
    model_memory,
    prompt_strategy: str = "fixed_prompt",
    action_mask: np.ndarray = None,
):

    act_seq = []
    trans_size = action_length + obs_length + 1
    for i_act in range(action_length):
        if i_act == 0 or model_memory is None:
            act_flag, pos_id = _get_action_flag_and_position_id(
                0, len(current_seq) - 1, obs_length, action_length, prepend_trans_num=0
            )
        else:
            pos_id = np.array([0])
        x = RLTaskInput(
            tensor_seq=current_seq,
            vision_seq=vision_seq,
            text_seq=None,
            attention_mask=None,
            loss_mask=None,
            label=None,
            position_id=torch.tensor(pos_id, dtype=torch.long),
        )
        x.to(device=model.device)
        x.apply(lambda x: x[None, ...])
        res = model([x], compute_loss=False, mems=model_memory)
        logits = res[0]
        if model_memory is not None:
            model_memory = res[-1]
            # if model_memory[0].shape[1] >= args.n_position:
            #     model_memory = truncate_memory(model_memory, obs_length, action_length)
        logits = masked_logits_for_action(
            args, logits, discrete_action, action_space, env_action_mask=action_mask
        )
        logits = logits[:, -1, :]
        # print(i_act, logits)
        preds = logits.argmax(-1)
        if model_memory is None:
            current_seq = torch.cat([current_seq, preds.cpu()], dim=0)
            if len(current_seq) > args.n_position:
                if args.use_prompt and prompt_strategy == "fixed_prompt":
                    window_seq_view = torch.roll(
                        current_seq[len_fixed_prompt:], -trans_size
                    )
                    current_seq[len_fixed_prompt:].data.copy_(window_seq_view.data)
                    current_seq = current_seq[:-trans_size]

                    if vision_seq is not None:
                        window_img_view = torch.roll(
                            vision_seq[len_fixed_prompt_img:], -1
                        )
                        vision_seq[len_fixed_prompt_img:].data.copy_(
                            window_img_view.data
                        )
                        vision_seq = vision_seq[:-1]
                else:
                    current_seq, vision_seq = truncate_sequence_by_stepsize(
                        current_seq, vision_seq, obs_length, action_length, None
                    )
            # if len(current_seq) > args.n_position:
            #     current_seq, vision_seq = truncate_sequence_by_stepsize(
            #         current_seq, vision_seq, obs_length, action_length, None)
        else:
            # although cpu() may have a new copy, prevent side effect
            # of recover_model_predict_token_to_tokenizer_raw where
            # there are some inplace operations
            # (ming): memory net uses moving prompt!
            assert prompt_strategy != "fixed_prompt"
            current_seq = preds.cpu().clone()
            vision_seq = None
        preds = recover_model_predict_token_to_tokenizer_raw(
            args, preds, discrete_action
        )
        act_seq.append(preds.cpu())

    # pass last dim of action to model to memorize
    if model_memory is not None:
        x = RLTaskInput(
            tensor_seq=current_seq,
            vision_seq=None,
            text_seq=None,
            attention_mask=None,
            loss_mask=None,
            label=None,
            position_id=torch.tensor([0], dtype=torch.long),
        )
        x.to(device=model.device)
        x.apply(lambda x: x[None, ...])

        _, _, model_memory = model([x], compute_loss=False, mems=model_memory)
        # if model_memory[0].shape[1] >= args.n_position:
        #     model_memory = truncate_memory(model_memory, obs_length, action_length)

    if not discrete_action:
        act = cont_tokenizer.decode(torch.cat(act_seq), is_action=True).numpy()
    else:
        act = act_seq[0].item()
    return act, (current_seq, vision_seq), model_memory


def get_obs_length(args, text_tokenizer, obs):
    def _compute_single_obs_dim(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if "str" in x.dtype.name:
            encoded_x = text_tokenizer.encode(x.tolist())
            return len(encoded_x)
        elif x.ndim == 3 and x.shape[0] == 3:
            c, h, w = x.shape
            return (h // args.vision_patch_size) * (w // args.vision_patch_size)
        else:
            return x.size

    dims = tree.flatten(tree.map_structure(_compute_single_obs_dim, obs))
    return sum(dims)


@torch.no_grad()
def evalute_one_episode(
    args: Namespace,
    model: torch.nn.Module,
    env_name: str,
    env,
    *,
    rl_dataset: Optional[RLFullDataset] = None,
    max_step_size: Optional[int] = None
):
    eval_prompt_strat = args.prompt_strategy.split(";")[-1]
    cont_tokenizer = ContinuousScalarTokenizer(
        args.num_continuous_bin, args.discretize_mu, args.discretize_M
    )

    obs = env.reset()
    obs_length = env.obs_length
    discrete_action = judge_discrete_space(env.action_space)
    if discrete_action:
        action_length = 1
    else:
        action_length = np.prod(env.action_space.shape)


    done = False

    spliter_token_id = (
        args.text_vocab_size + args.num_discrete_values + args.num_continuous_bin
    )
    if args.overlap_with_text:
        spliter_token_id -= args.num_discrete_values
    spliter_token = torch.tensor([spliter_token_id], dtype=torch.long)
    current_seq, current_img, action_mask = env.reset()

    # ============== prepending prompt =================
    if args.use_prompt:
        fixed_prompt, prepend_img = env.get_prompt(strict_length=args.strict_length, minimal_expert_data=args.minimal_expert_data)

        # truncate sequence by stepsize
        
        len_fixed_prompt = len(fixed_prompt)
        len_fixed_prompt_img = (
            len(prepend_img) if prepend_img is not None else 0
        )  # num_env x num_trans x c x h x w
        current_seq = torch.cat([fixed_prompt, current_seq, spliter_token])

        if prepend_img is not None:
            assert prepend_img.shape[1:] == current_img.shape[1:], (
                prepend_img.shape,
                current_img.shape,
            )
            current_img = torch.cat([prepend_img, current_img], dim=0)
    else:
        len_fixed_prompt = 0
        len_fixed_prompt_img = 0
    # ============== prepending prompt =================

    ##### START EVALUATION ###########
    model_memory = model.init_mem(batch_size=1)

    done = False
    episode_return, episode_length = 0, 0
    trans_size = obs_length + action_length + 1

    while not done:
        act, (current_seq, current_img), model_memory = get_action(
            args,
            model,
            current_seq,
            current_img,
            cont_tokenizer,
            len_fixed_prompt,
            len_fixed_prompt_img,
            obs_length,
            action_length,
            discrete_action,
            env.action_space,
            model_memory,
            eval_prompt_strat,
            action_mask,
        )
        new_seq, new_img, action_mask, reward, done, info = env.step(act)
        
        episode_return += reward
        episode_length += 1

        if (max_step_size is not None) and (episode_length >= max_step_size):
            break

        
        if model_memory is None:
            current_seq = torch.cat([current_seq, new_seq, spliter_token])
            if current_img is not None:
                current_img = torch.cat([current_img, new_img], dim=0)
            if len(current_seq) > args.n_position:
                if args.use_prompt and eval_prompt_strat == "fixed_prompt":
                    window_seq_view = torch.roll(
                        current_seq[len_fixed_prompt:], -trans_size
                    )
                    current_seq[len_fixed_prompt:].data.copy_(window_seq_view.data)
                    current_seq = current_seq[:-trans_size]

                    if current_img is not None:
                        window_img_view = torch.roll(
                            current_img[len_fixed_prompt_img:], -1
                        )
                        current_img[len_fixed_prompt_img:].data.copy_(
                            window_img_view.data
                        )
                        current_img = current_img[:-1]
                else:
                    current_seq, current_img = truncate_sequence_by_stepsize(
                        current_seq, current_img, obs_length, action_length, None
                    )
        else:
            current_seq = torch.cat([new_seq, spliter_token])
            current_img = new_img

    # if hasattr(env, "get_normalized_score"):
    #     ans_ret = env.get_normalized_score(episode_return)
    # else:
    #     ans_ret = episode_return
    ans_ret = episode_return

    return ans_ret, episode_length


def evaluate_env(args, model_engine, env_name):
    eval_prompt_strat = args.prompt_strategy.split(";")[-1]
    text_tokenizer = build_text_tokenizer(
        args.tokenizer_save_path, False, None, None, None,
    )
    cont_tokenizer = ContinuousScalarTokenizer(
        args.num_continuous_bin, args.discretize_mu, args.discretize_M
    )

    # Build RLFullDataset if not passed
    build_rl_ds_fn = functools.partial(
        RLFullDataset,
        tokenizers=[text_tokenizer, cont_tokenizer],
        overlap_with_text=args.overlap_with_text,
        num_discrete_values=args.num_discrete_values,
        prompt_ratio=args.prompt_ratio,
        prompt_prob=args.prompt_prob,
        prompt_at_final_transition_prob=args.prompt_at_final_transition_prob,
        mask_prompt_action_loss=args.mask_prompt_action_loss,
        use_prompt=args.use_prompt,
        prompt_strategy=args.prompt_strategy.split(";")[0],
        cache_path=args.rl_dataset_cache_dir,
        vision_patch_size=args.vision_patch_size,
    )

    env = LMPromptEnv(
        env_name, args.n_position, build_rl_ds_fn, eval_prompt_strat
    )
    ep_len, ep_r = 0, 0
    for _ in range(args.num_trials):
        r, l = evalute_one_episode(
            args, model_engine, env_name, env, max_step_size=args.max_step_size
        )
        ep_len += l / args.num_trials
        ep_r += r / args.num_trials
    print("Complete evaluate {}, ep_r: {}, ep_len: {}".format(env_name, ep_r, ep_len))
    return ep_r, ep_len


def parallel_evaluate_env(args, model_engine, env_names: List[str]):
    """Split env_names according to rank-world and then gather results on rank_0"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    indices = torch.arange(len(env_names)).chunk(world_size)
    if len(indices) > rank:
        indices = indices[rank].cpu().numpy().tolist()
        chosen_envs = [env_names[i] for i in indices]

        local_results = {
            env_name: evaluate_env(args, model_engine, env_name)
            for env_name in chosen_envs
        }
    else:
        local_results = None

    total_results = [None for _ in range(world_size)]

    dist.gather_object(
        local_results, 
        total_results if dist.get_rank() == 0 else None, 
        dst=0
    )
    res = dict()
    
    for _res in total_results:
        if _res is not None:
            res.update(_res)

    return res


def main(args):
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args)
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


    model_engine, _, _, _ = deepspeed.initialize(args, model, mpu=mpu,)
    if args.load_dir:
        load_path, client_state = model_engine.load_checkpoint(args.load_dir, args.ckpt_tag)
        print_rank_0("load model successfully.")
    model_engine.eval()

    env_res = parallel_evaluate_env(args, model_engine, args.env_name)
    for env_name, (ep_r, ep_l) in env_res.items():
        print_rank_0(
            "Environment {} test results: return :{}, length: {}.".format(
                env_name, ep_r, ep_l
            )
        )
        

    for task_suite_name in args.task_suite_name:
        task_module = importlib.import_module("d4rl.{}".format(task_suite_name))
        all_tasks = copy.deepcopy(task_module.ALL_ENVS)
        print_rank_0("Test for Task Suite {}, there are {} tasks ...".format(task_suite_name, len(all_tasks)))

        env_res = parallel_evaluate_env(args, model_engine, all_tasks)
        for env_name, (ep_r, ep_l) in env_res.items():
            print_rank_0(
                "\tEnvironment {}:{} test results: return :{}, length: {}.".format(
                    task_suite_name, env_name, ep_r, ep_l
                )
            )


if __name__ == "__main__":
    args = get_args()

    with torch.no_grad():
        main(args)
