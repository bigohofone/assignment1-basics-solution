python -m cs336_basics.solutions._train_bpe \
    --tokenizer_config_path ./cs336_basics/configs/tokenizer_config.json \
    --input_path ./data/TinyStoriesV2-GPT4-train.txt \
    --vocab_path ./out/ts_train/tokenizer/vocab.pkl \
    --merges_path ./out/ts_train/tokenizer/merges.pkl \
    --n_proc 8