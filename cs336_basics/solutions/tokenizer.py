import regex as re
import pickle
from typing import Iterable, Iterator
from itertools import pairwise
from .train_bpe import (
    GPT2_SPLIT_PATTERN,
    generate_sub_chunks,
    pretokenize
)


class Tokenizer():
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.pattern = re.compile(GPT2_SPLIT_PATTERN)
        
        self.idx_to_vocab = vocab
        self.vocab_to_idx = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_to_rank = {m: i for i, m in enumerate(merges)}
        
        if special_tokens:
            self.special_tokens = sorted([tok for tok in special_tokens], key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(map(re.escape, self.special_tokens)) + ")")
            self._update_vocab(self.special_tokens)
        else:
            self.special_tokens = None
            self.special_pattern = None
            
    def _update_vocab(
        self, 
        vocabs_to_update: list[str]
    ):
        _id = max(self.idx_to_vocab.keys()) + 1 
        for vocab in vocabs_to_update:
            vocab = vocab.encode('utf-8')
            if vocab in self.vocab_to_idx:
                continue
            self.idx_to_vocab[_id] = vocab
            self.vocab_to_idx[vocab] = _id
            _id += 1 

    @classmethod
    def from_files(
        cls, 
        vocab_path: str, 
        merges_path: str, 
        special_tokens: list[str] | None = None,
        **kwargs
    ) -> "Tokenizer":
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _bpe(self, tok: str) -> list[int]:
        word = [bytes([b]) for b in tok.encode('utf-8')]

        while True:
            min_rank = float('inf')
            pair_to_merge = None

            for pair in pairwise(word):
                rank = self.merge_to_rank.get(pair)
                
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    pair_to_merge = pair

            if pair_to_merge is None:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair_to_merge:
                    new_word.append(word[i] + word[i+1])
                    i += 2 
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word

        return [self.vocab_to_idx[token] for token in word]
        
    
    def encode(
        self,
        text: str
    ) -> list[int]:
        ids = []
        
        for sub_chunk, is_special_token in generate_sub_chunks(text, self.special_pattern):
            if is_special_token:
                ids.append(self.vocab_to_idx[sub_chunk.encode('utf-8')])
                continue
            
            for tok in pretokenize(sub_chunk, self.pattern):
                ids.extend(self._bpe(tok))
        
        return ids
                
    
    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(
        self,
        ids: list[int]
    ) -> str:
        text = b''.join([self.idx_to_vocab[_id] for _id in ids])
        return text.decode('utf-8', errors='ignore')