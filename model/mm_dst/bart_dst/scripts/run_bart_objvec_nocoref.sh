python run_bart_objvec_nocoref.py \
--add_special_tokens=../data_object_special/simmc_special_tokens.json \
--item2id=./item2id.json \
--train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
--train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
--eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
--eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
--output_dir=../save_nocoref/model \
--train_batch_size=12 \
--output_eval_file=../save_nocoref/model/report.txt \
--num_train_epochs=10  \
--eval_steps=1000  \
--warmup_steps=10000 \