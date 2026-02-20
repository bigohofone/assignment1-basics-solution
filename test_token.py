from cs336_basics.solutions.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files(
    "/home/aikusrv03/users/stanford-cs336-assignment-solutions/assignment1-basics/out/ts_valid/tokenizer/vocab.pkl",
    "/home/aikusrv03/users/stanford-cs336-assignment-solutions/assignment1-basics/out/ts_valid/tokenizer/merges.pkl",
    ["<|endoftext|>"]
)

fpath = '/home/aikusrv03/users/stanford-cs336-assignment-solutions/assignment1-basics/out/ts_valid/preprocessing/shard_w3.npy'

import numpy as np

f = np.load(fpath, mmap_mode='r')

print(f[:512])
print(tokenizer.decode(f[:512]))