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

"""Wrap a gym env to make its output suitable for gato"""

import gym
import d4rl
from typing import Optional, Union, Dict, Tuple, List
import torch
import numpy as np
import tree

from src.data.rl_dataset import RLFullDataset


class LMPromptEnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        sequence_length: int,
        build_dataset_fn: callable,
        eval_prompt_strat: str,
    ):
        self.env = gym.make(env_name)
        self.ds: RLFullDataset = build_dataset_fn(env_name, sequence_length)
        self.text_tokenizer = self.ds.text_tokenizer
        self.cont_tokenizer = self.ds.discretizer
        self.text_vocab_size = self.text_tokenizer.vocab_size
        self.num_discrete_values = self.ds.num_discrete_values
        self.overlap_with_text = self.ds.overlap_with_text
        self.num_continuous_bin = self.cont_tokenizer.num_continuous_bin
        self.vision_patch_size = self.ds.vision_patch_size
        self.eval_prompt_strat = eval_prompt_strat

        self.action_length = self.ds.action_dim
        self.obs_length = self.ds.observation_dim

        self.seq_length = sequence_length

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    @property
    def spliter_token_id(self):
        spliter_token_id = (
            self.text_vocab_size + self.num_discrete_values + self.num_continuous_bin
        )
        if self.overlap_with_text:
            spliter_token_id -= self.num_discrete_values
        return spliter_token_id

    def reset(self, return_with_prompt: bool = True):
        obs = self.env.reset()

        current_seq, current_img = self.build_rl_task_input(raw_obs=obs)

        return (
            current_seq,
            current_img,
            self.get_current_action_mask(),
        )

    def step(self, act):
        obs, reward, done, info = self.env.step(act)
        new_seq, new_img = self.build_rl_task_input(raw_obs=obs)

        return new_seq, new_img, self.get_current_action_mask(), reward, done, info

    def get_current_action_mask(self):
        return (
            self.env.get_cur_action_mask()
            if hasattr(self.env, "get_cur_action_mask")
            else None
        )

    def get_prompt(
        self, 
        strict_length: bool=True,
        minimal_expert_data: bool=False  # strict length for potential batch eval.
    ):
        spliter_token = torch.tensor([self.spliter_token_id], dtype=torch.long)
        
        encoded_demo: Dict[str, np.ndarray] = self.ds.sample_expert_demonstration(
            strategy=self.eval_prompt_strat, 
            strict_length=strict_length,
            sample_peak=(not minimal_expert_data)
        )
        prepend_obs, prepend_img = self.build_rl_task_input(
            o_text=encoded_demo["obs/text"],
            o_image=encoded_demo["obs/image"],
            o_tensor=encoded_demo["obs/tensor"],
        )
        # shape alignment
        if prepend_obs.ndim == 1:
            prepend_obs.unsqueeze_(0)
        prepend_act = (
            torch.from_numpy(encoded_demo["actions"])
            .long()
            .to(prepend_obs.device)
            .reshape(len(prepend_obs), -1)
        )
        # (n_batch, 1)
        batch_spliter_token = (
            torch.ones((prepend_obs.shape[0], 1)).long().to(prepend_obs.device)
            * self.spliter_token_id
        )

        fixed_prompt = torch.cat(
            [prepend_obs, batch_spliter_token, prepend_act], dim=-1
        )
        prepend_tensor = fixed_prompt.flatten().long()
        
        return prepend_tensor, prepend_img

    # discrete, ..., continuous
    def encode_obs(self, x, obs_dim=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        o_text, o_image, o_tensor = None, None, None
        if "str" in x.dtype.name:
            encoded = self.text_tokenizer(
                x.tolist(), padding='max_length', truncation=True, max_length=obs_dim
            )["input_ids"]
            o_text = np.array(encoded)
        elif x.ndim == 3:
            c, h, w = x.shape
            assert c == 3
            o_image = x
        elif "float" in x.dtype.name:
            x = self.cont_tokenizer.discretize(x, is_action=False)
            x = x + self.num_discrete_values + self.text_vocab_size
            if self.overlap_with_text:
                x = x - self.num_discrete_values
            o_tensor = x
        elif "int" in x.dtype.name:
            if x.ndim == 0:
                x = x[None]
            o_tensor = x
            if not self.overlap_with_text:
                o_tensor += self.text_vocab_size
        else:
            raise ValueError
        return o_text, o_image, o_tensor

    def build_rl_task_input(
        self,
        raw_obs: Union[Dict, np.ndarray] = None,
        o_text: Union[Dict, np.ndarray] = None,
        o_image: Union[Dict, np.ndarray] = None,
        o_tensor: Union[Dict, np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if raw_obs is not None:
            processed_obs = tree.map_structure(
                self.encode_obs, raw_obs, self.ds.observation_dims_for_spec
            )
            if isinstance(processed_obs, dict):
                o_text = {k: v[0] for k, v in processed_obs.items()}
                o_image = {k: v[1] for k, v in processed_obs.items()}
                o_tensor = {k: v[2] for k, v in processed_obs.items()}
            else:
                o_text, o_image, o_tensor = processed_obs

        res = []
        assert not (o_text is None and o_image is None and o_tensor is None)
        input_img = None
        if o_text is not None:
            if isinstance(o_text, dict):
                for k in sorted(o_text):
                    if o_text[k] is not None:
                        res.append(o_text[k])
            else:
                res.append(o_text[k])
        if isinstance(o_image, dict):
            o_image = [v for v in o_image.values() if v is not None]
            assert (
                len(o_image) <= 1
            ), "Currently We only support one image in observation"
            o_image = o_image[0] if len(o_image) > 0 else None
        if o_image is not None:
            if len(o_image.shape) == 4:
                b, c, h, w = o_image.shape
            else:
                b = 0  # for ignore
                c, h, w = o_image.shape
            p = self.vision_patch_size
            image_len = (h // p) * (w // p)
            if b == 0:
                res.append(np.ones((image_len,)) * -1)
                input_img = torch.tensor(o_image[None], dtype=torch.float32)
            else:
                res.append(np.ones((b, image_len)) * -1)
                input_img = torch.tensor(o_image, dtype=torch.float32)
        if o_tensor is not None:
            if isinstance(o_tensor, dict):
                for k in sorted(o_tensor):
                    if o_tensor[k] is not None:
                        res.append(o_tensor[k])
            else:
                res.append(o_tensor)
        input_tensor = torch.tensor(
            np.concatenate(res, axis=-1), dtype=torch.long
        ).squeeze()

        return input_tensor, input_img

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed) # for random choice