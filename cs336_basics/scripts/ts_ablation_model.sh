uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml 

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/ablation_model/remove_rope.yml

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/ablation_model/use_post_norm.yml

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/ablation_model/use_silu.yml

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/ablation_model/remove_rmsnorm.yml