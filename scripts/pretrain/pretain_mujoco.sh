#!/bin/bash
set -ex

export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

if [ $# -lt 1 ] ; then
    DS_PORT=29502
else
    DS_PORT=$1
fi


BASE_PATH=.

DATA_PATH="
    hopper-expert-v0 rl
"

EVAL_ENVS="
    hopper-expert-v0
"


DS_CONFIG=ds_config.json

TP=1
PP=1
NLAYERS=4
HIDDEN=256
NHEAD=4
SEQ_LEN=256

GLOBAL_BATCH=256
MICRO_BATCH=256

USE_PROMPT="True"
PROMPT_RATIO=0.5
PROMPT_STRATEGY="stochastic_subseq;moving_prompt"

OUTPUT_DIR=results/nl${NLAYERS}_hs${HIDDEN}_l${SEQ_LEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_ml${MEM_LEN}_p${USE_PROMPT}_pr${PROMPT_RATIO}
mkdir -p $OUTPUT_DIR

RL_CACHE_DIR=[your data cache dir]

cat <<EOT > $DS_CONFIG
{
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "steps_per_print": 1000,
    
    "fp16": {
        "enabled": true,
        "initial_scale_power": 12
    },


    "tensorboard": {
        "enabled": true,
        "output_path": "$OUTPUT_DIR/ds/",
        "job_name": "train_all"
    },
    "wall_clock_breakdown" : false
}
EOT



export NCCL_DEBUG=warn

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"


deepspeed --include "localhost:0" --master_port $DS_PORT src/pretrain/pretrain_multimodal.py \
    --deepspeed_port $DS_PORT \
    --model "transformer_xl" \
    --n-layer $NLAYERS \
    --n-embed $HIDDEN \
    --n-head $NHEAD \
    --n-position $SEQ_LEN \
    --mem-len $SEQ_LEN \
    --pre-lnorm False \
    --untie-r False \
    --share-input-output-embedding True \
    --same-length True \
    --activation-fn "geglu" \
    --use-deepnorm False \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 500000 \
    --eval-iters 3 \
    --eval-interval 5000 \
    --data-path $DATA_PATH \
    --tensorboard-dir $OUTPUT_DIR \
    --save-dir $OUTPUT_DIR \
    --save-interval 10000 \
    --seed 42 \
    --lr-decay-style "cosine" \
    --lr-warmup-iters 5000 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --optimizer "adamw" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --tokenizer-save-path "my_tokenizer" \
    --data-impl "mmap" \
    --eval-env-names $EVAL_ENVS --minimal-expert-data False \
    --num-workers 2 \
    --fp16 True \
    --use-prompt $USE_PROMPT \
    --prompt-strategy $PROMPT_STRATEGY \
    --prompt-ratio $PROMPT_RATIO \
    --prompt-at-final-transition-prob 0.5 \
    --rl-dataset-cache-dir $RL_CACHE_DIR \
    --eval-ic-iter 0 --eval-vqa-iter 0 \
    --dataloader-type "cyclic" \
    $ds_args | tee ${OUTPUT_DIR}/output.log


 
