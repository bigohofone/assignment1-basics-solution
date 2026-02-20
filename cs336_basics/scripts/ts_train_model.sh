deepspeed --num_gpus=8 _train_model.py \
    --ds_config_path ./configs/ds_config.json \
    --model_config_path ./configs/model_config.json \
    --dataset_dir ./out/ts_train \
    --checkpoint_dir ./out/ts_train/checkpoints \
    --save_interval 2000 \
    --lr 6e-4 \
    --context_length 256 \
    --total_iters 10000