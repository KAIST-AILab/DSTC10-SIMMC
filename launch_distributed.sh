python -m torch.distributed.launch \
    --nproc_per_node=2 \
    tasks/disambiguate/trainer.py \
        --config_name roberta-base \
        --validate_ratio=1.0 \
        --batch_size=4 \
        --gradient_accumulation_steps=8 \
        --max_epochs=20 \
        --max_turns=3 \
        # --warmup_ratio=0.06 \
        --learning_rate=2e-5 \
        --use_special_tokens \
        --seed=42 \
        --fp16 \
        --dropout=0.1 \
        --workers=0