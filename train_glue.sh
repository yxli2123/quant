python run_glue.py \
    --task_name mnli \
    --model_name_or_path khalidalt/DeBERTa-v3-large-mnli \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --output_dir ./output/ \
    --seed 42 \
    --with_tracking \
    --num_bits 8 \
    --reduced_rank 8 \
    --act_quant \
    --svd_init \
    --clip_val 0.2 \

