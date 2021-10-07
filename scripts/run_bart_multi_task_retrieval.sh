python run_bart_multi_task_retrieval.py \
    --path_output=retrieval_result.json \
    --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
    --candidate_file=../data_object_special/simmc2_dials_dstc10_devtest_retrieval.json \
    --model_dir=../checkpoint-23000 \
    --item2id item2id.json \
    --add_special_tokens=../data_object_special/simmc_special_tokens.json