import os
import glob
import random
import numpy as np
import deepspeed
import torch


def load_dataset(dataset_dir):
    pattern = os.path.join(dataset_dir, "*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard_w*.npy files found in {dataset_dir}")
    
    arrays = [np.load(f) for f in files]
    full_dataset = np.concatenate(arrays)
    return full_dataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_dist(config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    wsz = deepspeed.comm.get_world_size()
    rank = deepspeed.comm.get_rank()
    
    seed = config.training.get('seed', 42) + rank
    seed_everything(seed)
    
    return rank, wsz


def fuse_qkv_state_dict(state_dict):
    new_state_dict = state_dict.copy()
    processed_prefixes = set()

    for key in list(state_dict.keys()):
        if ".q_proj.weight" in key:
            prefix = key.rsplit(".q_proj.weight", 1)[0]
            
            if prefix in processed_prefixes:
                continue
                
            q_w = state_dict[f"{prefix}.q_proj.weight"]
            k_w = state_dict[f"{prefix}.k_proj.weight"]
            v_w = state_dict[f"{prefix}.v_proj.weight"]
            new_state_dict[f"{prefix}.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            
            if f"{prefix}.q_proj.bias" in state_dict:
                q_b = state_dict[f"{prefix}.q_proj.bias"]
                k_b = state_dict[f"{prefix}.k_proj.bias"]
                v_b = state_dict[f"{prefix}.v_proj.bias"]
                new_state_dict[f"{prefix}.qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
            
            for suffix in [".q_proj.weight", ".k_proj.weight", ".v_proj.weight", 
                           ".q_proj.bias", ".k_proj.bias", ".v_proj.bias"]:
                full_key = prefix + suffix
                if full_key in new_state_dict:
                    del new_state_dict[full_key]
            
            processed_prefixes.add(prefix)

    return new_state_dict