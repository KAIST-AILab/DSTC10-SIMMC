#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Train (multi-modal)
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m bart_simmc1_dst.scripts.run_bart \
    --output_dir="${PATH_DIR}"/bart_simmc1_dst/save/model \
    --model_type=facebook/bart-base \
    --model_name_or_path=facebook/bart-base \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/bart_simmc1_dst/data/simmc2_special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/bart_simmc1_dst/data/simmc2_dials_dstc10_train_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/bart_simmc1_dst/data/simmc2_dials_dstc10_dev_target.txt \
    --num_train_epochs=2 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \


