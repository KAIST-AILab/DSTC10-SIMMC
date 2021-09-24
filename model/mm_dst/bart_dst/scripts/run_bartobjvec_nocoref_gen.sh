 python run_bartobjvec_generation_nocoref.py \
    --model_dir=../save_nocoref/model/checkpoint-22000 \
    --stop_token='<EOS>' \
    --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
    --path_output=../results_nocoref/simmc2_dials_dstc10_devtest_predicted.txt \
    --item2id=./item2id.json 
