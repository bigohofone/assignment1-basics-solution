import os
import json
import argparse
from .train_bpe import train_bpe, SPLIT_SPECIAL_TOKEN


parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer_config_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--vocab_path', type=str)
parser.add_argument('--merges_path', type=str)
parser.add_argument('--n_proc', type=int, default=4)
args = parser.parse_args()


with open(args.tokenizer_config_path, 'r', encoding='utf-8') as f:
    tokenizer_cfg = json.load(f)

output_dir = os.path.dirname(args.vocab_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

train_bpe(
    input_path=args.input_path,
    vocab_size=tokenizer_cfg['vocab_size'],
    special_tokens=tokenizer_cfg['special_tokens'],
    vocab_path=args.vocab_path,
    merges_path=args.merges_path,
    n_proc=args.n_proc
)
