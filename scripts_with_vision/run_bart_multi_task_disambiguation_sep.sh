python run_bart_multi_task_disambiguation_sep.py \
    --path_output=../devtest_results_sep/disambiguation_result_sep.json \
    --prompts_from_file=../data_object_special_sep/simmc2_dials_dstc10_devtest_predict.txt \
    --disambiguation_file=../data_object_special_sep/simmc2_dials_dstc10_devtest_inference_disambiguation.json \
    --model_dir=../multi_task_sep/model/checkpoint-29000 \
    --item2id item2id_sep.json \
    --add_special_tokens=../data_object_special_sep/simmc_special_tokens.json