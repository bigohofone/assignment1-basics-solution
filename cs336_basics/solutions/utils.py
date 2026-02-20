import torch

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