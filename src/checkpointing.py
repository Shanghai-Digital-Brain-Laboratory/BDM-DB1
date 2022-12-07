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

"""Checkpoint functionality"""
import deepspeed

def save_checkpoint(args, iteration, model_engine: deepspeed.DeepSpeedEngine):
    client_state = {}
    client_state["args"] = args
    client_state["iteration"] = iteration

    model_engine.save_checkpoint(args.save_dir, client_state=client_state, tag="latest_model")
