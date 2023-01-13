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

"""Configuration for RL task evaluation"""

from src.config import str2bool

def _add_rl_eval_args(parser):
    group = parser.add_argument_group(title="rl evaluation")
    group.add_argument("--num-trials", type=int, default=3)
    group.add_argument("--max-step-size", type=int, default=None)
    group.add_argument("--strict-length", type=str2bool, default=True)
    group.add_argument("--minimal-expert-data", type=str2bool, required=True)
    return parser