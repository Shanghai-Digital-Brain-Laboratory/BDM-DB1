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

"""Implementation of continous scalar tokenizer as Described in [Gato][https://www.deepmind.com/publications/a-generalist-agent]"""


import numpy as np
import torch

class ContinuousScalarTokenizer:
    def __init__(
        self, num_continuous_bin: int = 1024, mu: float = 100.0, M: float = 256.0
    ):
        self.num_continuous_bin = num_continuous_bin
        self.mu = mu
        self.M = M

    def discretize(self, x, is_action: bool):
        """
        Discretization of float scalars, if is_action then don't need mu-law scaling.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()

        if not is_action:
            x_mu_lawed = (
                torch.sign(x)
                * torch.log(torch.abs(x) * self.mu + 1.0)
                / torch.log(torch.tensor(self.mu * self.M + 1.0))
            )
            x = torch.clamp(x_mu_lawed, -1, 1)

        x = ((x + 1) / 2 * self.num_continuous_bin).int()
        x = torch.clamp(x, 0, self.num_continuous_bin - 1).int()
        return x

    def decode(self, x, is_action: bool):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.max() >= self.num_continuous_bin or x.min() < 0:
            print(
                "Warning of exceeded range of discrete number to recontruct, "
                "by default values will be cliped, min: {}, max:{}".format(
                    x.min(), x.max()
                )
            )
            x = np.clip(x, 0, self.num_continuous_bin - 1)

        x = (x.float() / self.num_continuous_bin) * 2 - 1
        if not is_action:
            x = torch.sign(x) * ((1 + self.M * self.mu) ** torch.abs(x) - 1) / self.mu

        return x
