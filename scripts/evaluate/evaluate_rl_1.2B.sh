#!/bin/bash
set -ex

if [ $# -lt 1 ] ; then
    DS_PORT=29503
else
    DS_PORT=$1
fi

TAG_NAME="db1_870task_checkpoint"

DS_CONFIG=ds_config.json

TP=1
PP=1
NLAYERS=24
HIDDEN=2048
NHEAD=16
SEQ_LEN=1024

OUTPUT_DIR=rl_eval_results/${TAG_NAME}
mkdir -p $OUTPUT_DIR

USE_PROMPT="True"
PROMPT_RATIO=0.5
PROMPT_STRATEGY="stochastic_subseq;moving_prompt"

GLOBAL_BATCH=512
MICRO_BATCH=4

cat <<EOT > $DS_CONFIG
{
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "steps_per_print": 1,
    "fp16": {
        "enabled": true,
        "initial_scale_power": 12
    },
    "wall_clock_breakdown" : true
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"

TEST_ENVS=""
TEST_SUITES="
    babyai
    gym_sokoban
    metaworld
    modular_rl
    atari
    dmc
    gym_procgen
    dmlab
"


RL_CACHE_DIR=[your rl_minimal_exp_data]

deepspeed --include localhost --master_port $DS_PORT src/evaluation/evaluate_rl.py \
    --load-dir [your model_checkpoint] \
    --ckpt-tag $TAG_NAME \
    --model "transformer_xl" \
    --deepspeed_port $DS_PORT \
    --untie-r False \
    --share-input-output-embedding True \
    --same-length True \
    --pre-lnorm False \
    --n-layer $NLAYERS \
    --n-embed $HIDDEN \
    --n-head $NHEAD \
    --n-position $SEQ_LEN \
    --env-name $TEST_ENVS \
    --task-suite-name $TEST_SUITES \
    --tokenizer-save-path "my_tokenizer" \
    --fp16 True \
    --mem-len $SEQ_LEN \
    --activation-fn "geglu" \
    --use-prompt $USE_PROMPT \
    --prompt-strategy $PROMPT_STRATEGY \
    --prompt-ratio $PROMPT_RATIO \
    --prompt-at-final-transition-prob 0.5 \
    --rl-dataset-cache-dir $RL_CACHE_DIR \
    --minimal-expert-data True \
    --num-trials 5 \
    $ds_args | tee "${OUTPUT_DIR}/results.output"

