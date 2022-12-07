![img_v2_4a7d4460-005b-4ab9-a316-472f873ec93g](https://user-images.githubusercontent.com/25078430/201826696-dea2fd8c-c643-4d93-813f-a2179ab4e779.png)

# Digital Brain 1 (DB1): A large-scale multi-modal multi-task pre-trained decision model

This repo implements a multi-modal Transformer - DB1, which is pretrained with multiple tasks, including natural language modeling, image caption, and single agent decision-making tasks (such as pixel input video games, continuous control, and TSP problems).

DB1 is also a reproduction of GATO and achieves similar performance on all of the tasks mentioned above.
Specifically, on 76% of all 870 simulated decision making tasks, DB1 achieves $\ge$ 50% expert performance.

<img src='https://user-images.githubusercontent.com/31499806/206094245-fabe405f-8e28-4741-b02d-60d1757820d3.gif' height='150px'> <img src='https://user-images.githubusercontent.com/31499806/206094261-d69889ee-d05e-4d3c-95ec-48280156089b.gif' height='150px'> 
<img src='https://user-images.githubusercontent.com/31499806/206095468-6347fd1b-a110-4ad7-aac5-7c432f6343d9.gif' height='150px'> 
<img src='https://user-images.githubusercontent.com/31499806/206095425-a6e3144c-e58f-4136-a2ba-fbf20f1c3ba9.gif' height='150px'> 
<img src='https://user-images.githubusercontent.com/31499806/206094808-0f1ad57f-aa81-4a06-91f5-86e2ca49bca8.gif' height='150px'> 
<img src='https://user-images.githubusercontent.com/31499806/206095739-55407905-2547-40cf-95c5-49ac52697c51.gif' height='150px'> 

Pretraining scripts, model checkpoints, and training data will come soon.

## Environment setup

Suppose you've already installed cuda and nvidia-drivers successfully.

Download files from this [site](https://onedrive.live.com/?authkey=%21aic%5f24u3bhfesta&id=1f59bb5f1b4d1366%212401&cid=1f59bb5f1b4d1366). 
there are:
- DB1's model checkpoint `db1_870task_checkpoint/`
- Python libraries to install `external_libs.tar.gz`
- Minimal data for evaluation `minimal_expert_data.tar.gz`
```

conda create -n db1 python=3.9 -y
conda activate db1

# use the version compatible with your environments.
conda install pytorch=1.12.1 cudatoolkit=11.3 -c pytorch -y 
pip install -r requirements.txt

sudo apt update && sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libglew-dev patchelf gcc -y

pip install 'gym[atari]'
autorom

# mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar xzvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco && mv mujoco210 ~/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
pip3 install -U 'mujoco-py<2.2,>=2.1'

# D4RL
tar xzvf external_libs.tar.gz

pip install -e d4rl
pip install -e metaworld

git clone https://github.com/digital-brain-sh/mycocoevalcap.git
pip install -e mycocoevalcap

# Dataset index building functions
cd src/data
make
cd -

```
### DMLab installation

```
sudo apt-get install pkg-config g++ zlib1g-dev unzip libsdl2-2.0 libffi-dev gettext freeglut3-dev libsdl2-dev python3 zip libosmesa6-dev python-dev python-numpy python-pil python-enum34 python3-dev python3-numpy python3-pil
```

Install [bazel](https://docs.bazel.build/versions/main/install-ubuntu.html#install-with-installer-ubuntu).

Clone our DMLAB repo and build it.
```
git clone https://github.com/digital-brain-sh/lab
cd lab
bazel build --cxxopt="--std=c++14" -c opt --python_version=PY3 //python/pip_package:build_pip_package --verbose_failures'
# suppose you have already downloaded the dmlab package we provided
# or you can build the package with `./bazel-bin/python/pip_package/build_pip_package ~/`
pip install deepmind_lab-1.0-py3-none-any.whl
```


Follow the instructions [here](https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008) to download additional data of brady_konkle_oliva2008 for deepmind lab. In our code, default path of the data is at `/raid/brady_konkle_oliva2008`. If you wish to change the path, move the downloaded files into and then set `$DMLAB_DATASET_PATH` to `[your dir]/brady_konle_oliva2008`.

## Data preparation
### Model checkpoints
Download DB1's model checkpoint `db1_870task_checkpoint` from this [site](https://onedrive.live.com/?authkey=%21AIc%5F24u3BHFesTA&id=1F59BB5F1B4D1366%212401&cid=1F59BB5F1B4D1366). 
```
mkdir model_checkpoints
mv db1_870task_checkpoint model_checkpoints
```
### Minimal expert demonstration dataset for RL evaluation.
Currently we only provide a minimal RL dataset as expert demonstration for evaluation.
We build a minimal RL dataset acting as prompt during evaluation.
```
tar xzvf minimal_expert_data.tar.gz
# you will get a folder named `rl_minimal_exp_data`
```
## Runnable scripts
### Evaluation
#### Simulated decision making tasks
Fill in `$RL_CACHE_DIR` in the following script and argument of `--load-dir [your model_checkpoint]`, for other potential checkpoints dir, you can change `$TAG_NAME` to load them, for detail you can see [DeepSpeed Model Checkpoint](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html).

```
cd [DB1's directory]
export PYTHONPATH=. # you can also use absolute path or any correct form.
sh scripts/evaluate_rl_1.3B.sh [choose a port for deepspeed, e.g. 29002]
```

Then the performance on all environments will be recorded in the log at `rl_eval_results/db1_870task_checkpoint/results.output`

#### Comming soon.
- [ ] Minimal data for running TSP problems.
- [ ] Text generation and Image Caption scripts will be released soon.
- [ ] Finetuning results
- [ ] Pretrained models with modern tricks like DeepNorm and etc.

## Implementation details

### Framework Overview

We adapt our training procedure and preprocessing logic for NLP tasks and vision tasks with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). To improve the data-loading efficiency, we equip data caching and lazy load for large-scale datasets. For the implementation of [TransformerXL](https://arxiv.org/abs/1901.02860) and its relative positional encoding and memory caching, we reference TransformerXL. In addition, extra techniques and tricks were taken into consideration to stabilize the training procedure.


### Training Overview
We use [DeepSpeed](https://www.deepspeed.ai/) to speedup the training process and scale our models. However, since DB1 is designed for tasks cross multiple modalities, we find it difficult to apply modern techniques like tensor/pipeline model parallelism. So we only use distributed data parallel during pretraining.

Currently we provide you with the checkpoint of our pretrained 1.2B model, we've tested it on DGX A100 for pretraining and evaluation, and RTX 3090 for evaluation. 


<!-- ## Acknowledgement -->

## Contact
If you have any questions about this repo, feel free to leave an issue.
## Join Us
Get Interested in our project? Or have great passions in:
1. Multi-Agent Learning and Game AI
2. Operation Research and Optimization
3. Robotics and Control
4. Visual and Graphic Intelligence
5. Data Mining and so on

Welcome! Why not take a look at https://digitalbrain.cn/talents?

With the leading scientists, enginneers and field experts, we are going to provide **Better Decisions for Better World**!

### Digital Brain Laboratory
Digital Brain Laboratory, Shanghai, is co-founded by the founding partner and chairman of CMC Captital, Mr. Ruigang Li, and world-renowned scientist in the field of decision intelligence, Prof. Jun Wang.

### Recruitment

<img src='https://user-images.githubusercontent.com/25078430/201830084-ebb731db-9a84-4e37-b6e1-7dbb34bc8fc1.png' width='150px'>

### Recruitment for Students & Internships

<img src='https://user-images.githubusercontent.com/25078430/201830117-5ff5daf0-df66-4eee-bf82-109838d42e17.png' width='150px'>
