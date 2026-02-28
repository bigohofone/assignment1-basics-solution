import os
import json
import yaml
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.solutions.tokenizer import Tokenizer
from cs336_basics.solutions.transformer_lm import TransformerLM
from cs336_basics.solutions.decoding import decode


parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--input_text", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, default='./checkpoints')
parser.add_argument("--ckpt_tag", type=str, default='final')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_dir = os.path.join(args.ckpt_dir, args.run_name)
ckpt_path = os.path.join(ckpt_dir, args.ckpt_tag, "mp_rank_00_model_states.pt")

with open(os.path.join(ckpt_dir, 'model.yaml'), 'r') as f:
    model_config = yaml.safe_load(f)
    
model = TransformerLM(**model_config)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['module'])
model.eval()

with open(os.path.join(ckpt_dir, 'tokenizer.yaml'), 'r') as f:
    tokenizer_config = yaml.safe_load(f)
    
tokenizer = Tokenizer.from_files(
    vocab_path=os.path.join(ckpt_dir, 'vocab.pkl'), 
    merges_path=os.path.join(ckpt_dir, 'merges.pkl'), 
    **tokenizer_config
)

prompt = "Once upon a time"
generated_text = decode(model, tokenizer, prompt, max_new_tokens=128)

print(f"{prompt} -> {generated_text}")