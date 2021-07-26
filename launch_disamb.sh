# PREPROCESS DATA FOR DISAMBIGUATION

# mkdir -p data/disambiguate
# python baseline/disambiguate/format_disambiguation_data.py \
#     --simmc_train_json data/simmc2_dials_dstc10_train.json \
#     --simmc_dev_json data/simmc2_dials_dstc10_dev.json \
#     --simmc_devtest_json data/simmc2_dials_dstc10_devtest.json \
#     --disambiguate_save_path data/disambiguate

# BASELINE MODEL FROM AUTHORS

# python baseline/disambiguate/train_model.py \
#     --train_file data/disambiguate/simmc2_disambiguate_dstc10_train.json \
#     --dev_file data/disambiguate/simmc2_disambiguate_dstc10_dev.json \
#     --devtest_file data/disambiguate/simmc2_disambiguate_dstc10_devtest.json \
#     --use_gpu \
#     --batch_size=8 \
#     --learning_rate=2e-5 \
#     --max_turns=5

# MODIFIED CODE FROM AUTHORS TO RUN BERT

# python baseline/disambiguate/train_bert.py \
#     --train_file data/disambiguate/simmc2_disambiguate_dstc10_train.json \
#     --dev_file data/disambiguate/simmc2_disambiguate_dstc10_dev.json \
#     --devtest_file data/disambiguate/simmc2_disambiguate_dstc10_devtest.json \
#     --use_gpu \
#     --batch_size=8 \
#     --learning_rate=2e-5 \
#     --max_turns=5 \
#     --model_name bert-base-cased \
#     --fp16 \
#     --warmup_steps=500

# OUR IMPLEMENTATION TO RUN BERT W/ DDP SUPPORT

CUDA_VISIBLE_DEVICES=$1 python tasks/disambiguate/trainer.py \
    --config_name=$2 \
    --validate_ratio=1.0 \
    --batch_size=4 \
    --gradient_accumulation_steps=16\
    --max_epochs=20 \
    --max_turns=3 \
    --warmup_ratio=0.06 \
    --learning_rate=2e-5
    --workers=0 \
    --use_special_tokens \
    --seed=42 \
    --dropout=0.2 \
    --fp16 \
    --flooding=0.001