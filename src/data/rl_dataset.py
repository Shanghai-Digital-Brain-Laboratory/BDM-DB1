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

"""RL dataset, take reference from Trajectory transformer, support data caching and prompt generation"""

import os
from pathlib import Path
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union, Tuple
from warnings import warn

import numpy as np
from src.mpu import print_rank_0
from src.data.blendable_dataset import BlendableDataset

from src.data.gpt_dataset import GPTDataset
import gym
import pickle as pkl
from src.data.input_specs import RLTaskInput

from src.tokenizer.scalar_tokenizer import ContinuousScalarTokenizer
import torch
import pdb
import d4rl
import math
import importlib
import tree

# this is default value, you can configure it outside by "--rl-dataset-cache-dir {your path}"
CACHE_DIR = "/nfs/dgx10/raid/ziyu_rl_data_cache"


def _get_action_flag_and_position_id(
    index_l, index_r, obs_seq_len, act_seq_len, prepend_trans_num
):
    # This method can be used only if index_l is left aligned to the start of a timestep
    seq_length = index_r - index_l + 1
    action_flag_res = np.zeros((seq_length,), dtype=np.int64)
    position_id_res = np.zeros_like(action_flag_res)
    step_size = obs_seq_len + act_seq_len + 1
    prepend_mask_length = prepend_trans_num * step_size

    if prepend_mask_length > 0:
        action_flag_res[:prepend_mask_length] = 0

    # NOTE(ming): we currently do not distinguish the position id for prompt sequence
    for i in range(0, seq_length, step_size):
        position_id_res[i : i + obs_seq_len + 1] = 1 + np.arange(
            min(obs_seq_len + 1, seq_length - i)
        )
    for i in range(prepend_mask_length, seq_length, step_size):
        # if seq_length - i < step_size:
        #     break
        action_flag_res[
            i + obs_seq_len + 1 : min(seq_length, i + step_size)
        ] = 1  # np.ones(
        #     (act_seq_len,)
        # )  # Hard shape check

    return action_flag_res, position_id_res


def qlearning_dataset_with_timeouts(
    env, dataset=None, terminate_on_end=False, **kwargs
):
    """Reference from TT code, but we do not need next_obs"""
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    obs_ = deepcopy(tree.map_structure(lambda x: x, dataset["observations"]))
    action_ = dataset["actions"].copy()
    reward_ = dataset["rewards"].copy()
    terminal_done_ = dataset["terminals"].copy()
    if "timeouts" in dataset:
        timeout_done_ = dataset["timeouts"].copy()
        done_ = terminal_done_ | timeout_done_
    else:
        done_ = terminal_done_

    return {
        "observations": obs_,
        "actions": action_,
        "rewards": reward_[:, None],
        "terminals": done_[:, None],
        "realterminals": terminal_done_[:, None],
    }


def segment(traj_input, terminals, max_path_length=None):
    """Segment data as a list of trajectory arrays
    traj_input: a tree-structure of np.ndarray
        whose first axes are of the same length
    terminals: sigals to split traj_input into lists
    max_path_length: default None, whether to truncate data
    when trajectory is too long.
    """
    if max_path_length is not None:
        assert max_path_length > 0

    data_size = set(tree.flatten(tree.map_structure(lambda x: len(x), traj_input)))

    assert len(data_size) == 1
    data_size = list(data_size)[0]
    assert data_size == len(terminals)

    # tmp_traj = []
    trajectories = []
    start = 0
    for i, term in enumerate(terminals):
        if term.squeeze() or (
            max_path_length is not None and i - start + 1 >= max_path_length
        ):
            trajectories.append(
                tree.map_structure(lambda x: x[start : i + 1], traj_input)
            )
            start = i + 1
    if start < i + 1:
        trajectories.append(tree.map_structure(lambda x: x[start : i + 1], traj_input))
    return trajectories


class RLFullDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_name,
        seq_length: int,
        tokenizers: List,
        overlap_with_text: bool = True,
        num_discrete_values: int = 1024,
        prompt_ratio: float = 0.5,
        prompt_prob: float = 0.25,
        prompt_at_final_transition_prob: float = 0.5,
        mask_prompt_action_loss: bool = True,
        vision_patch_size: int = 16,
        is_lazy: bool = True,
        cache_path: str = CACHE_DIR,
        use_prompt: bool = True,
        prompt_strategy: str = "stochastic_subseq",
    ):
        """Current implementation of RL dataset
        env_name: str, the environment name passed to d4rl gym.make, should be first registered
        seq_length: the length of output sequence, as the sum of prompt_length + predicted_length
        tokenizers: a list of text and continuous tokenizers
        num_transition_prompt: number of transition to be prepended as prompt
        prompt_prob: the probability of prepending prompt
        prompt_at_final_transition_prob: the probability of use the end of an episode as prompt
        mask_prompt_action_loss: whether to mask the action_loss of prompt
        """
        # print_rank_0("Building full RL dataset of {}....".format(env_name))
        self.env = gym.make(env_name)
        self.name = env_name
        # XXX(ming): do not use fixed path length, read it from env_spec instead.
        # XXX(ziyu): after discussion we think here we do not set a limit of trajectory length.
        self.max_path_length = None
        self.output_sequence_length = seq_length
        self.prompt_strategy = prompt_strategy
        self.use_prompt = use_prompt

        self.vision_patch_size = vision_patch_size
        self.prompt_prob = prompt_prob
        self.prompt_at_final_transition_prob = prompt_at_final_transition_prob
        self.prompt_ratio = prompt_ratio
        self.mask_prompt_action_loss = mask_prompt_action_loss

        self.text_tokenizer, self.discretizer = tokenizers
        self.num_discrete_values = num_discrete_values
        self.overlap_with_text = overlap_with_text

        self.is_lazy = is_lazy
        self.cache_path = Path(cache_path) / env_name
        self.obs_path = self.cache_path / "observations"
        self.act_path = self.cache_path / "actions"
        self.reward_path = self.cache_path / "rewards"
        self.meta_path = self.cache_path / "meta"
        self.index_path = self.meta_path / f"indices_{seq_length}.npy"
        self.cached = self.is_cached()

        if not self.cached:
            # dataset = self.env.get_dataset()
            dataset = qlearning_dataset_with_timeouts(
                self.env.unwrapped, terminate_on_end=True
            )

            observations = dataset["observations"]
            actions = dataset["actions"]
            terminals = dataset["terminals"]
            rewards = dataset["rewards"]

            res = segment(
                (observations, actions, rewards), terminals, self.max_path_length
            )

            self.observations, self.actions, self.rewards = tuple(zip(*res))
            # XXX(ming): use returns for high-quality traj selection
            self.traj_returns = np.asarray(
                [e.sum() for e in self.rewards], dtype=np.float32
            )

            # since reward is definitely an array
            self.path_lengths = [len(x) for x in self.rewards]

            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    self.cache_data()
            else:
                self.cache_data()

            tmp_obs, tmp_act = self.get_obs_action_by_path_idx(0)
            self.obs_type_spec = self.get_obs_type_spec(tmp_obs)
            self.observation_dims_for_spec = self.get_observation_dim(tmp_obs)
            self.observation_dim = sum(tree.flatten(self.observation_dims_for_spec))
            self.action_dim = self.get_action_dim(tmp_act[0])
            trans_dim = self.observation_dim + self.action_dim

            # we need sample (seq_len + 1) to split token and label
            # self.transition_num = math.ceil((self.output_sequence_length+1) / (trans_dim+1))
            # compute ceil of transition num
            self.transition_num = (self.output_sequence_length + trans_dim) // (
                trans_dim + 1
            )
            self.prompt_transition_num = int(prompt_ratio * self.transition_num)
            self.predicted_transition_num = (
                self.transition_num - self.prompt_transition_num
            )

            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    self.cache_meta_data()
            else:
                self.cache_meta_data()
        else:
            self.path_lengths = np.load(self.cache_path / "path_lengths.npy")
            self.traj_returns = np.load(self.cache_path / "traj_returns.npy")
            self.load_cache()
            assert self.output_sequence_length == seq_length

        if self.index_path.exists():
            self.indices = np.load(self.index_path, allow_pickle=True, mmap_mode="r")
        else:
            ## get valid indices
            # indices = []
            # for path_ind, length in enumerate(self.path_lengths):
            #     end = length - 1
            #     for i in range(end):
            #         real_end = min(i + self.transition_num, length)
            #         indices.append((path_ind, i, real_end))
            # self.indices = np.array(indices)

            ## Another possible index
            # indices = []
            # for path_ind, length in enumerate(self.path_lengths):
            #     end = length
            #     for i in range(0, end, max(1, self.transition_num - 1)):
            #         real_end = min(i + self.transition_num, length)
            #         indices.append((path_ind, i, real_end))
            #     else:
            #         if i < length:
            #             indices.append((path_ind, i, length))
            # self.indices = np.array(indices)

            from src.data import helpers

            # XXX(ziyu): I only use int32_t instead of int64_t
            cpp_idx = helpers.build_rl_sample_idx(
                self.path_lengths, self.transition_num
            )
            self.indices = np.array(cpp_idx)
            np.save(self.index_path, np.array(self.indices))

        # XXX(ming): we select prompt according to the length, why don't we select the top 10% rewarded-traj?
        self.traj_idx_ret_tuples = sorted(
            [(i, self.traj_returns[i]) for i in range(len(self.path_lengths))],
            key=lambda x: x[1],
            reverse=True,
        )

    def is_cached(self):
        return (
            os.path.exists(self.obs_path)
            and os.path.exists(self.act_path)
            and os.path.exists(self.reward_path)
            # XXX: path_length file exists
            # and os.path.exists(self.meta_path)
        )

    def is_meta_cached(self):
        return os.path.exists(self.meta_path)

    def cache_data(self):
        os.makedirs(self.obs_path, exist_ok=True)
        os.makedirs(self.act_path, exist_ok=True)
        os.makedirs(self.reward_path, exist_ok=True)
        tree.map_structure_with_path(
            lambda p, unused: os.makedirs(self.obs_path / "/".join(p), exist_ok=True),
            self.observations[0],
        )

        # # save data for lazy = false
        # np.save(self.obs_path + os.sep + "observations.npy" , np.array(self.observations))
        # np.save(self.act_path + os.sep + "actions.npy" , np.array(self.actions))
        # np.save(self.reward_path + os.sep + "rewards.npy" , np.array(self.rewards))

        # save data for lazy = true
        n_traj = len(self.rewards)

        for i in range(n_traj):
            # np.save(
            #     self.obs_path / (str(i) + ".npy"), np.array(self.observations[i])
            # )
            tree.map_structure_with_path(
                lambda p, x: np.save(self.obs_path / "/".join(p) / f"{i}.npy", x),
                self.observations[i],
            )
            np.save(
                self.act_path / (str(i) + ".npy"),
                np.array(self.actions[i], dtype=self.actions[i].dtype),
            )
            np.save(
                self.reward_path / (str(i) + ".npy"),
                np.array(self.rewards[i], dtype=self.rewards[i].dtype),
            )

        np.save(self.cache_path / "path_lengths.npy", np.array(self.path_lengths))
        np.save(self.cache_path / "traj_returns.npy", self.traj_returns)

    def cache_meta_data(self):
        # save meta info
        os.makedirs(os.path.join(self.meta_path), exist_ok=True)

        np.save(
            self.meta_path / "output_sequence_length.npy",
            np.array(self.output_sequence_length),
        )
        np.save(self.meta_path / "obs_type_spec.npy", np.array(self.obs_type_spec))
        np.save(
            self.meta_path / "observation_dims_for_spec.npy",
            np.array(self.observation_dims_for_spec),
        )
        np.save(
            self.meta_path / "observation_dim.npy", np.array(self.observation_dim),
        )
        np.save(self.meta_path / "action_dim.npy", np.array(self.action_dim))
        np.save(
            self.meta_path / "transition_sequence_length.npy",
            np.array(self.transition_num),
        )

    def load_cache(self):
        # load data
        if not self.is_lazy:
            raise RuntimeError
            # self.observations = np.load(self.obs_path + os.sep + "observations.npy", allow_pickle=True)
            # self.actions = np.load(self.act_path + os.sep + "actions.npy", allow_pickle=True)
            # self.rewards = np.load(self.reward_path + os.sep + "rewards.npy", allow_pickle=True)

        # load meta info
        self.output_sequence_length = np.load(
            self.meta_path / "output_sequence_length.npy", allow_pickle=True
        )
        self.obs_type_spec = np.load(
            self.meta_path / "obs_type_spec.npy", allow_pickle=True
        ).item()
        self.observation_dims_for_spec = np.load(
            self.meta_path / "observation_dims_for_spec.npy", allow_pickle=True
        ).item()
        self.observation_dim = np.load(
            self.meta_path / "observation_dim.npy", allow_pickle=True
        )
        self.action_dim = np.load(self.meta_path / "action_dim.npy", allow_pickle=True)
        self.transition_num = np.load(
            self.meta_path / "transition_sequence_length.npy", allow_pickle=True,
        )
        self.prompt_transition_num = int(self.prompt_ratio * self.transition_num)
        self.predicted_transition_num = self.transition_num - self.prompt_transition_num

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.get(idx, with_raw=False)

    def postprocess_obs_and_act(
        self, obs_array: Union[Dict, np.ndarray], act_array: np.ndarray
    ) -> Tuple[Tuple["Text", "Image", "Tensor"], "Actions"]:  # type: ignore pylance
        """Process observations and actions
        Observation: Suppose two forms, a single ndarray or a dict of ndarray
            if float number, do discretize.
            if image input, doesn't change
            if text, do tokenization
        Action:
            if float number, do discretize

        return: (text=None, image=None, tensor=None), act
        """
        if hasattr(self.env, "post_process_fn"):
            obs_array = self.env.post_process_fn(obs_array)

        if hasattr(self.env, "action_mapper"):
            act_array = self.env.action_mapper(act_array)

        n_disc = self.num_discrete_values

        def postprocess_obs(obs_array, obs_type, obs_dim):
            o_text, o_image, o_tensor = None, None, None
            if obs_type == "text":
                o_text = self.text_tokenizer(
                    obs_array.tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=obs_dim,
                )["input_ids"]
                o_text = np.array(o_text, dtype=np.int32)
            elif obs_type == "image":
                o_image = obs_array
            elif obs_type == "float":
                obs_array = self.discretizer.discretize(
                    obs_array, is_action=False
                ).numpy()
                o_tensor = obs_array + n_disc + self.text_tokenizer.vocab_size
                if self.overlap_with_text:
                    o_tensor = o_tensor - n_disc
            elif obs_type == "discrete":
                assert (
                    obs_array.min() >= 0 and obs_array.max() < self.num_discrete_values
                )
                o_tensor = obs_array
                if not self.overlap_with_text:
                    o_tensor = o_tensor + self.text_tokenizer.vocab_size
            if o_tensor is not None and o_tensor.ndim < 2:
                o_tensor = o_tensor[:, None]
            return o_text, o_image, o_tensor

        processed_obs = tree.map_structure(
            postprocess_obs,
            obs_array,
            self.obs_type_spec,
            self.observation_dims_for_spec,
        )

        # XXX: can I use tree to do this?
        if isinstance(processed_obs, dict):
            o_text = {k: v[0] for k, v in processed_obs.items()}
            o_image = {k: v[1] for k, v in processed_obs.items()}
            o_tensor = {k: v[2] for k, v in processed_obs.items()}
        else:
            o_text, o_image, o_tensor = processed_obs

        if "float" in act_array.dtype.name:
            act_array = self.discretizer.discretize(act_array, is_action=True).numpy()
            n_disc = self.num_discrete_values
            processed_act = act_array + n_disc + self.text_tokenizer.vocab_size
            if self.overlap_with_text:
                processed_act = processed_act - n_disc
        else:
            assert act_array.min() >= 0 and act_array.max() < self.num_discrete_values
            if act_array.ndim == 1:
                act_array = act_array[:, None]
            processed_act = act_array
            if not self.overlap_with_text:
                processed_act = processed_act + self.text_tokenizer.vocab_size

        return (o_text, o_image, o_tensor), processed_act

    def prepend_prompt(
        self, path_idx: int, observations: Union[Dict, np.ndarray], actions: np.ndarray
    ) -> Tuple[Union[Dict, np.ndarray], np.ndarray]:
        """Prepending prompt to observations and actions.

        Args:
            path_idx (int): Path index for searching trajectory to prepend prompt.
            observations (Union[Dict, np.ndarray]): Original observations.
            actions (np.ndarray): Original actions sequence.

        Returns:
            Tuple[Union[Dict, np.ndarray], np.ndarray]: A tuple of processed observation and action sequences.
        """

        if isinstance(observations, Dict):
            for k, v in observations.items():
                assert self.transition_num, (
                    k,
                    len(v),
                    self.predicted_transition_num,
                    self.transition_num,
                )
        else:
            assert len(observations) <= self.transition_num, (
                len(observations),
                self.predicted_transition_num,
                self.transition_num,
            )

        real_prepend_trans_num = 0
        if path_idx >= 0 and np.random.random() < self.prompt_prob:
            obs_traj, action_traj = self.get_obs_action_by_path_idx(path_idx)
            path_length = self.path_lengths[path_idx]
            # GATO 1.0 strategy
            # real_prepend_trans_num = self.prompt_transition_num
            if np.random.random() < self.prompt_at_final_transition_prob:
                # prompt as goal
                trans_obs = tree.map_structure(
                    lambda x: x[-self.prompt_transition_num :], obs_traj
                )
                trans_act = action_traj[-self.prompt_transition_num :]
            else:
                if self.prompt_strategy == "stochastic_timestep":
                    # prompt as exploration
                    random_idx = np.random.choice(
                        path_length, self.prompt_transition_num, replace=False
                    )
                    random_idx.sort()
                    trans_obs = tree.map_structure(lambda x: x[random_idx], obs_traj)
                    trans_act = action_traj[random_idx]
                else:  # stochastic_subseq
                    random_start = np.random.choice(
                        max(path_length - self.prompt_transition_num, 1)
                    )
                    random_end = random_start + self.prompt_transition_num
                    trans_obs = tree.map_structure(
                        lambda x: x[random_start:random_end], obs_traj
                    )
                    trans_act = action_traj[random_start:random_end]

            real_prepend_trans_num = len(trans_act)

            # clip original actions and observations.
            offset_range = max(0, len(actions) - self.predicted_transition_num)
            offset = (
                np.random.choice(offset_range) if offset_range > 0 else offset_range
            )

            observations = tree.map_structure(
                lambda x: x[offset : offset + self.predicted_transition_num],
                observations,
            )
            obs_holder = tree.map_structure(
                lambda x, y: np.zeros(
                    (x.shape[0] + y.shape[0],) + x.shape[1:], dtype=x.dtype
                ),
                observations,
                trans_obs,
            )
            if isinstance(observations, Dict):
                for k, v in obs_holder.items():
                    v[:real_prepend_trans_num] = trans_obs[k]
                    v[
                        real_prepend_trans_num : real_prepend_trans_num
                        + observations[k].shape[0]
                    ] = observations[k]
            else:
                obs_holder[:real_prepend_trans_num] = trans_obs
                obs_holder[real_prepend_trans_num:] = observations
            # observations = tree.map_structure(lambda x, y: np.concatenate([x, y], axis=0), trans_obs, observations)

            actions = actions[offset : offset + self.predicted_transition_num]
            act_holder = np.zeros(
                (trans_act.shape[0] + actions.shape[0],) + actions.shape[1:],
                dtype=actions.dtype,
            )
            act_holder[:real_prepend_trans_num] = trans_act
            act_holder[real_prepend_trans_num:] = actions
            # actions = np.concatenate([trans_act, actions], axis=0)
        else:
            obs_holder = observations
            act_holder = actions

        return obs_holder, act_holder, real_prepend_trans_num

    def get_obs_action_by_path_idx(
        self, path_ind: int, start_ind: int = None, end_ind: int = None
    ):
        # here for lazy loading
        start_ind = 0 if start_ind is None else start_ind
        if self.cached and self.is_lazy:
            lazy_actions = np.load(
                self.act_path / (str(path_ind) + ".npy"), mmap_mode="r"
            )

            end_ind = end_ind or len(lazy_actions)

            # lazy_obs = np.load(
            #     self.obs_path / (str(path_ind) + ".npy")
            # )
            lazy_obs = tree.map_structure_with_path(
                lambda p, unused: np.load(
                    self.obs_path / "/".join(p) / f"{path_ind}.npy", mmap_mode="r"
                ),
                self.obs_type_spec,
            )
            # if lazy_obs.dtype.name == "object":
            #     lazy_obs = lazy_obs.item()
            actions = lazy_actions[start_ind:end_ind]
            observations = tree.map_structure(lambda x: x[start_ind:end_ind], lazy_obs)
        else:
            end_ind = end_ind or len(self.actions[path_ind])
            actions = self.actions[path_ind][start_ind:end_ind]
            observations = tree.map_structure(
                lambda x: x[start_ind:end_ind], self.observations[path_ind]
            )

        return observations, actions

    def get(self, idx: int, with_raw=False):
        if idx >= len(self.indices):
            # warn("input index exceeded {} >= {}".format(
            #     idx, len(self.indices)))
            idx = idx % len(self.indices)
        n_cont = self.discretizer.num_continuous_bin
        n_disc = self.num_discrete_values
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]

        # here for lazy loading
        observations, actions = self.get_obs_action_by_path_idx(
            path_ind, start_ind, end_ind
        )

        # random select a trajectory for prepending
        if self.use_prompt:
            rand_path_idx = np.random.choice(len(self.path_lengths))
            observations, actions, real_prepend_trans_num = self.prepend_prompt(
                rand_path_idx, observations, actions
            )
        else:
            real_prepend_trans_num = 0

        (o_text, o_image, o_tensor), act_discrete = self.postprocess_obs_and_act(
            observations, actions
        )

        obs_discrete = []
        #### Processing text ##########
        if o_text is not None:
            # XXX(ziyu): since we assume the structure of observation will only be
            # either an array or a dict of arrays, we can just write some simple codes
            if isinstance(o_text, dict):
                for k in sorted(o_text):
                    if o_text[k] is not None:
                        obs_discrete.append(o_text[k])
            else:
                obs_discrete.append(o_text)

        #### Processing Image ##########
        if isinstance(o_image, dict):
            o_image = [v for v in o_image.values() if v is not None]
            assert (
                len(o_image) <= 1
            ), "Currently We only support one image in observation"
            o_image = o_image[0] if len(o_image) > 0 else None
        if o_image is not None:
            n, c, h, w = o_image.shape
            p = self.vision_patch_size
            image_len = (h // p) * (w // p)
            # if n < self.transition_num:
            if n < self.transition_num:
                pad_o_image = np.zeros((self.transition_num, c, h, w), dtype=np.float32)
                pad_o_image[:n] = o_image
                o_image = pad_o_image
            obs_discrete.append(np.ones((n, image_len)) * -1)

        #### Processing tensor ##########
        if o_tensor is not None:
            if isinstance(o_tensor, dict):
                for k in sorted(o_tensor):
                    if o_tensor[k] is not None:
                        obs_discrete.append(o_tensor[k])
            else:
                obs_discrete.append(o_tensor)
        obs_discrete = np.concatenate(obs_discrete, axis=1)
        # assert self.observation_dim == obs_discrete.shape[1]

        spliter_token_id = n_cont + self.text_tokenizer.vocab_size
        if not self.overlap_with_text:
            spliter_token_id += n_disc

        joined_discrete = np.concatenate(
            [
                obs_discrete,  # use -1 as temperary padding for vision inputs
                spliter_token_id
                * np.ones((act_discrete.shape[0], 1)),  # This is seperator
                act_discrete,
            ],
            axis=1,
        )

        joined_discrete = joined_discrete.flatten().astype(np.int64)
        action_flag, position_id = _get_action_flag_and_position_id(
            0,
            len(joined_discrete) - 1,
            self.observation_dim,
            self.action_dim,
            real_prepend_trans_num,
        )
        transision_dim = self.observation_dim + self.action_dim + 1
        if end_ind > path_length:
            action_flag[(path_length - start_ind) * transision_dim :] = 0

        # truncate to self.output_length+1

        target_seq_len = self.output_sequence_length + 1
        position_id = _truncate_or_pad_to_match_seq_len(position_id, target_seq_len)
        action_flag = _truncate_or_pad_to_match_seq_len(action_flag, target_seq_len)
        joined_discrete = _truncate_or_pad_to_match_seq_len(
            joined_discrete, target_seq_len
        )

        if o_image is not None and o_image.shape[0] > act_discrete.shape[0]:
            for i in range(act_discrete.shape[0], o_image.shape[0]):
                joined_discrete[
                    i
                    * transision_dim : min(
                        target_seq_len, i * transision_dim + self.observation_dim
                    )
                ] = -1

        if hasattr(self.env, "build_task_input"):
            res = self.env.build_task_input(
                position_id=position_id[:-1],
                attention_mask=None,
                text_seq=None,
                vision_seq=o_image,
                tensor_seq=joined_discrete[:-1],
                loss_mask=action_flag[1:],
                label=joined_discrete[1:],
            )
        else:
            res = RLTaskInput(
                position_id=position_id[:-1],
                attention_mask=None,
                text_seq=None,
                vision_seq=o_image,
                tensor_seq=joined_discrete[:-1],
                loss_mask=action_flag[1:],
                label=joined_discrete[1:],
            )
        res.apply(lambda x: torch.tensor(x))
        res.apply(lambda x: x[None, ...])
        if with_raw:
            return res, (observations, actions)
        else:
            return res

    def get_observation_dim(self, obs):
        """Get the length of observation when feed into transformer
        The input observation is just one-timestep obs with shape
        (*obs_shape). e.g. (3, 84, 84)
        PS: Now we only support the case that the text of fixed length in
        an environment
        """
        if hasattr(self.env, "post_process_fn"):
            obs = self.env.post_process_fn(obs)

        def _compute_single_obs_dim(x):
            if "str" in x.dtype.name:
                encoded_x = self.text_tokenizer(x.tolist())["input_ids"]
                return max(len(tt) for tt in encoded_x)
            elif x.ndim == 4 and x.shape[1] == 3:
                b, c, h, w = x.shape
                return (h // self.vision_patch_size) * (w // self.vision_patch_size)
            else:
                return x[0].size

        dims = tree.map_structure(_compute_single_obs_dim, obs)
        return dims

    def get_action_dim(self, act):
        """Get the length of observation when feed into transformer
        The input observation is just one-timestep obs with shape
        """
        if hasattr(self.env, "action_mapper"):
            act = self.env.action_mapper(act)
        return act.shape[0] if len(act.shape) == 1 else 1

    def get_obs_type_spec(self, obs):
        """
        type=[text, image, float, discrete]
        """
        if hasattr(self.env, "post_process_fn"):
            obs = self.env.post_process_fn(obs)

        def _get_obs_type(x):
            if x.ndim == 4:
                assert (
                    x.shape[1] == 3
                ), "We assume the rgb input should of shape (3, h, w)"
                return "image"
            elif "float" in x.dtype.name:
                return "float"
            elif "str" in x.dtype.name:
                return "text"
            elif "int" in x.dtype.name:
                return "discrete"
            else:
                raise ValueError

        return tree.map_structure(_get_obs_type, obs)

    def sample_expert_demonstration(
        self, strategy: str, strict_length: bool, sample_peak: bool
    ) -> Dict[str, np.ndarray]:
        """Sample an expert demonstration and encode it as a dict {'actions', 'obs/text', 'obs/image', 'obs/tensor'}
        Returns:
            Dict[str, np.ndarray]: A dict of encoded demonstration.
        """

        # only first n_trans is need
        prompt_length = (
            self.prompt_transition_num
            if strategy == "fixed_prompt"
            else self.transition_num  # XXX(ziyu): remove "- 1" since I need all env has the same length of input
        )
        if sample_peak:
            # select demonstration from the top 10%
            stop_idx = int(len(self.traj_idx_ret_tuples) * 0.1)
            candidates = [x[0] for x in self.traj_idx_ret_tuples[:stop_idx]]
        else:
            candidates = np.arange(len(self.path_lengths))
        path_idx = np.random.choice(candidates)
        observation_traj, action_traj = self.get_obs_action_by_path_idx(path_idx)

        if strict_length:
            current_length = len(action_traj)
            obs_list, act_list = [observation_traj], [action_traj]
            while current_length < prompt_length:
                path_idx = np.random.choice(candidates)
                observation_traj, action_traj = self.get_obs_action_by_path_idx(
                    path_idx
                )
                obs_list.append(observation_traj)
                act_list.append(action_traj)
                current_length += len(action_traj)
            observation_traj = tree.map_structure(
                lambda *xs: np.concatenate(xs, axis=0), *obs_list
            )
            action_traj = np.concatenate(act_list, axis=0)

        actions = action_traj[:prompt_length]
        observations = tree.map_structure(lambda x: x[:prompt_length], observation_traj)

        ((o_text, o_image, o_tensor), act_discrete,) = self.postprocess_obs_and_act(
            observations, actions
        )

        # append parsed obs

        return {
            "actions": act_discrete,
            "obs/text": o_text,
            "obs/image": o_image,
            "obs/tensor": o_tensor,
        }


def _truncate_or_pad_to_match_seq_len(arr: np.ndarray, seq_len: int):
    if len(arr) > seq_len:
        return arr[:seq_len]
    elif len(arr) < seq_len:
        # by default it will pad constant zero
        return np.pad(arr, (0, seq_len - len(arr)))
    else:
        return arr


class RLTaskSuiteDataset(BlendableDataset):
    def __init__(
        self, env_name: str, seq_length: int, build_rl_full_dataset_fn: Callable
    ):
        task_module = importlib.import_module("d4rl.{}".format(env_name))
        all_tasks = deepcopy(task_module.ALL_ENVS)
        datasets = [
            build_rl_full_dataset_fn(task_name, seq_length) for task_name in all_tasks
        ]
        super().__init__(datasets, [1.0] * len(all_tasks))
        print_rank_0(
            "Create Dataset for {} task suite, there are {} tasks.".format(
                env_name, len(all_tasks)
            )
        )


class RLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        unused_name,
        unused_data_prefix,
        documents: np.ndarray,  # the name of "documents" is the same as GPTDataset
        underlying_dataset: Union[RLFullDataset, RLTaskSuiteDataset],
        unused_train_valid_test_num_samples,
        unused_seq_legth,
        unused_seed,
    ):
        """Build a subset of RLFullDataset
        documents: a np.ndarray which indicates the indicies in this dataset
            that should be a subset of underlying_dataset's indices
        underlying_dataset: the underlying RLFullDataset, which should
            be created earlier.
        """

        self.dataset = underlying_dataset
        assert documents.ndim == 1
        assert documents.min() >= 0 and documents.max() < len(underlying_dataset)

        self.indices = documents

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            # warn("input index exceeded {} >= {}".format(
            #     idx, len(self.indices)))
            idx = idx % len(self.indices)
        return self.dataset[self.indices[idx]]


class RLFinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self, env_name, seq_length, build_rl_full_dataset_fn: Callable, num_fewshot_episodes: Optional[int] = None
    ):
        super().__init__()
        self.ds = build_rl_full_dataset_fn(env_name, seq_length)
        self.num_fewshot = num_fewshot_episodes

        if self.num_fewshot is not None:
            selected_path_idx = np.random.choice(
                len(self.ds.path_lengths), size=self.num_fewshot, replace=False
            )
            self.selected_path_idx = np.sort(selected_path_idx)

            self.num_item_selected_path = (
                self.ds.path_lengths[self.selected_path_idx] - 1
            )
            self.size = np.sum(self.num_item_selected_path)

            offset = [0] + self.num_item_selected_path[:-1].tolist()
            self.selected_path_offset = np.array(offset)

            selected_path_offset_underlying = (
                self.ds.path_lengths.cumsum() - np.arange(len(self.ds.path_lengths)) - 1
            )
            self.selected_path_offset_underlying = selected_path_offset_underlying[
                self.selected_path_idx
            ]

    def __getitem__(self, idx):
        if self.num_fewshot is None:
            return self.ds[idx]
        else:
            path_idx = np.argwhere(idx >= self.selected_path_offset).max()
            inner_path_offset = idx - self.selected_path_offset[path_idx]
            underlying_idx = (
                self.selected_path_offset_underlying[path_idx] + inner_path_offset
            )
            return self.ds[underlying_idx]

    def __len__(self):
        if self.num_fewshot is None:
            return len(self.ds)
        else:
            return self.size


if __name__ == "__main__":
    from src.tokenizer.text_tokenizer import build_text_tokenizer

    start_time = time.time()
    # fullset = RLFullDataset(
    #     "hopper-expert-v0",
    #     100,
    #     tokenizers=(build_text_tokenizer("/home/ziyu/workspace/CDMA/benchmarking/Gato/my_tokenizer"),
    #         ContinuousScalarTokenizer(1024)),
    #     num_discrete_values=1024,
    # )
    build_rl_full_dataset_fn = lambda name, seq_len: RLFullDataset(
        name,
        seq_len,
        tokenizers=(
            build_text_tokenizer(
                "/home/ziyu/workspace/CDMA/benchmarking/Gato/my_tokenizer"
            ),
            ContinuousScalarTokenizer(1024),
        ),
        num_discrete_values=1024,
        # cache_path="/raid/ziyu_rl_data_cache_debug_lessdata",
    )
    # ds1 = build_rl_full_dataset_fn('tsp200-city-28-expert-v1', 1024)
    # import pdb; pdb.set_trace()
    # ds1[1]
    # print(ds1.path_lengths.max(), ds1.path_lengths.min(), ds1.path_lengths.mean())
    # print(ds1.traj_returns.max(), ds1.traj_returns.min(), ds1.traj_returns.mean())
    # pdb.set_trace()
    import functools
    fn = functools.partial(
        RLFullDataset,
        seq_length=1024,
        tokenizers=(
            build_text_tokenizer(
                "/home/ziyu/workspace/CDMA/benchmarking/Gato/my_tokenizer"
            ),
            ContinuousScalarTokenizer(1024),
        ),
        num_discrete_values=1024,
        cache_path="/nfs/dgx03/raid/ziyu_rl_data_cache",
    )
    # ds2 = RLFinetuneDataset("walker_7_main-expert-v0", fn, 1)
    # print(len(ds2))

    def build_env(env_name):
        env = fn(env_name)
        return None

    from concurrent.futures.process import ProcessPoolExecutor
    all_env_names = []
    # all_env_names.extend(d4rl.d4rl.atari.ALL_ENVS)
    # all_env_names.extend(d4rl.d4rl.babyai.ALL_ENVS)
    # all_env_names.extend(d4rl.d4rl.modular_rl.ALL_ENVS)
    all_env_names.extend(d4rl.d4rl.tsp.ALL_ENVS)
    all_env_names = [x for x in all_env_names if "city" in x]
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = pool.map(build_env, all_env_names)
        print("START")
        [_ for _ in res]
    print("YES")

    ds0 = RLTaskSuiteDataset("atari", 1024, build_rl_full_dataset_fn)
    # ds1 = RLTaskSuiteDataset("babyai", 1024, build_rl_full_dataset_fn)
    # ds2 = RLTaskSuiteDataset("modular_rl", 1024, build_rl_full_dataset_fn)
    # ds3 = RLTaskSuiteDataset("gym_procgen", 1024, build_rl_full_dataset_fn)
    # ds4 = RLTaskSuiteDataset("dmlab", 1024, build_rl_full_dataset_fn)
    # ds5 = RLTaskSuiteDataset("metaworld", 1024, build_rl_full_dataset_fn)
    # ds6 = RLTaskSuiteDataset("tsp", 1024, build_rl_full_dataset_fn)
    # ds7 = RLTaskSuiteDataset("dmc", 1024, build_rl_full_dataset_fn)
    # cnt = 0
    # while True:
    #     # i = np.random.choice(len(ds3))
    #     i = cnt
    #     ds3[i]
    #     cnt += 1
    #     print(cnt, end="\r")
    # import time
    # t0 = time.time()
    # for _ in range(100):
    #     idx = np.random.choice(len(ds1))
    #     ds1[idx]
    # t1 = time.time()
    # print("Random access time: {:.6}".format(t1 - t0))

    # ds = BlendableDataset([ds0, ds1, ds2], [1.0] * 3)
    # print(len(ds))
    # print("test: ", fullset[1000])
    # end_time = time.time()
    # print("run time: {}.".format(end_time-start_time))

    # from src.data.dataset_utils import get_train_valid_test_split_
    # splits = get_train_valid_test_split_("900,75,25", len(fullset))
    # ranges = [np.arange(splits[i], splits[i+1]) for i in range(len(splits)-1)]

    # train_ds = RLDataset(0, 0, ranges[0], fullset, 0,0,0)
    # # valid_ds = RLDataset(ranges[1], fullset)
    # # test_ds = RLDataset(ranges[2], fullset)
    # train_ds[0]

    # task_suites = [
    #     "atari",
    #     "babyai",
    #     "metaworld",
    #     "gym_procgen",
    #     "modular_rl",
    #     "gym_sokoban",
    #     "dmlab",
    #     "dmc",
    #     "tsp",
    # ]

    # dss = {
    #     k: RLTaskSuiteDataset(k, 1024, build_rl_full_dataset_fn) for k in task_suites
    # }
    # num_tokens_per_suite = {}
    # for k, ds in dss.items():
    #     num_tasks = len(ds.datasets)
    #     num_traj_per_task = {
    #         sub_ds.name: sub_ds.traj_returns.shape[0] for sub_ds in ds.datasets
    #     }
    #     num_total_traj = sum(num_traj_per_task.values())
    #     num_of_timestep_per_task = {
    #         sub_ds.name: sub_ds.path_lengths.sum() for sub_ds in ds.datasets
    #     }
    #     num_of_tokens_per_task = {
    #         sub_ds.name: num_of_timestep_per_task[sub_ds.name]
    #         * (sub_ds.observation_dim + sub_ds.action_dim + 1)
    #         for sub_ds in ds.datasets
    #     }
    #     num_total_tokens = sum(num_of_tokens_per_task.values())
    #     num_total_timestep = sum(num_of_timestep_per_task.values())
    #     num_tokens_per_suite[k] = num_total_tokens
    #     print(
    #         "task suite: {}, num of tasks: {}, num of total trajectories: {}, num of total timesteps: {}, num of total tokens: {}".format(
    #             k, num_tasks, num_total_traj, num_total_timestep, num_total_tokens
    #         )
    #     )
    #     print(
    #         "| {} | {} | {} | {} | {} |".format(
    #             k, num_tasks, num_total_traj, num_total_timestep, num_total_tokens
    #         )
    #     )
    
    # print(num_tokens_per_suite)
    # tmp = np.array(list(num_tokens_per_suite.values()))
    # tmp = tmp / tmp.sum() * 100

    # for i, k in enumerate(num_tokens_per_suite):
    #     print("{}: {:.4}".format(k, tmp[i]))
    
    # import pdb; pdb.set_trace()
