#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi


# Generate sentences (Furniture, multi-modal)
CUDA_VISIBLE_DEVICES=1 python -m bart_simmc1_dst.scripts.run_generation \
    --model_type=bart \
    --model_name_or_path="${PATH_DIR}"/bart_simmc1_dst/save/model/checkpoint-6000  \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/bart_simmc1_dst/data/simmc2_dials_dstc10_devtest_predict.txt \
    --path_output="${PATH_DIR}"/bart_simmc1_dst/results/simmc2_dials_dstc10_devtest_predicted.txt

