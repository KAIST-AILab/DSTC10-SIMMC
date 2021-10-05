 python run_bart_multi_task_mm_dst.py \
    --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
    --path_output=mm_dst_result.txt \
    --item2id=./item2id.json \
    --add_special_tokens=/home/yschoi/DSTC10-SIMMC/model/mm_dst/bart_dst/data_object_special/simmc_special_tokens.json \
    --batch_size=24 \
    --model_dir=../multi_task/model/checkpoint-20000 