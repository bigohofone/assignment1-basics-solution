import multiprocessing as mp
from collections import Counter
from functools import partial
import regex as re
from itertools import pairwise, chain
import heapq
import os
from typing import BinaryIO
import heapq
import pickle
from dataclasses import dataclass
from typing import Iterator
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_SPECIAL_TOKEN = "<|endoftext|>"


@dataclass(frozen=True, slots=True)
class ReverseLexicalPair:
    pair: tuple[bytes, bytes]

    def __lt__(self, other: "ReverseLexicalPair") -> bool:
        return self.pair > other.pair


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def generate_sub_chunks(
    text: str, 
    pattern: re.Pattern[str] | None
) -> Iterator[tuple[str, bool]]:
    """
    Yields segments of text, separating pattern matches from non-matches.
    
    Yields:
        (substring, is_match): 'is_match' is True if the substring matches the pattern.
    """
    if not pattern:
        yield text, False
        return

    last_end = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        
        # Yield non-matching text preceding the current match
        if start > last_end:
            yield text[last_end:start], False
        
        # Yield the match itself
        yield text[start:end], True
        last_end = end
    
    # Yield any remaining text after the last match
    if last_end < len(text):
        yield text[last_end:], False


def pretokenize(text: str, pattern: re.Pattern[str]) -> Iterator[str]:
    """Yields all substrings that match the given pattern."""
    return (match.group() for match in pattern.finditer(text))


def _count_words_worker(
    chunk: str, 
    pattern: re.Pattern[str], 
    special_pattern: re.Pattern[str]
) -> Counter:
    counts = Counter()
    
    for sub_chunk, is_special in generate_sub_chunks(chunk, special_pattern):
        if is_special:
            continue
        for word in pretokenize(sub_chunk, pattern):
            byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
            counts[byte_tuple] += 1
            
    return counts


def count_words(
    fpath: str, 
    special_tokens: list[str], 
    max_n_proc: int = 8
) -> Counter:
    n_proc = min(mp.cpu_count(), max_n_proc)
    
    with open(fpath, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_proc, SPLIT_SPECIAL_TOKEN.encode('utf-8'))
        chunks = []
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            f.seek(s)
            chunks.append(f.read(e - s).decode('utf-8'))

    pattern = re.compile(GPT2_SPLIT_PATTERN)
    if special_tokens:
        s_toks = sorted(special_tokens, key=len, reverse=True)
        special_tokens = [re.escape(tok) for tok in s_toks]
        special_pattern = re.compile("(" + "|".join(special_tokens) + ")")
    else:
        special_pattern = None
    
    global_cntr = Counter()
    
    worker_fn = partial(_count_words_worker, pattern=pattern, special_pattern=special_pattern)
    
    with mp.Pool(processes=n_proc) as pool:
        results = tqdm(
            pool.imap_unordered(worker_fn, chunks), 
            total=len(chunks), 
            desc="Counting words (Chunks)"
        )
        
        for result_counter in results:
            global_cntr.update(result_counter)
        
    return global_cntr


def _get_initial_stats(
    cntr: Counter
) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes], set[tuple[bytes]]]]:
    pair_stats = dict()
    pair_to_words = dict()
    for ts, c in cntr.items():
        for pair in pairwise(ts):
            pair_stats[pair] = pair_stats.get(pair, 0) + c
            pair_to_words.setdefault(pair, set()).add(ts)
            
    heap = [(-count, ReverseLexicalPair(pair), pair) for pair, count in pair_stats.items()] # To I need to sort by alphabetical order?
    heapq.heapify(heap)
    
    return pair_stats, pair_to_words, heap


def _merge(
    word_cntr: Counter, 
    pair_stats: dict[tuple[bytes], int], 
    pair_to_words: dict[tuple[bytes], tuple[bytes]], 
    heap: list[bytes], 
    vocab: dict[int, bytes], 
    merges: list[tuple[bytes]], 
    tok_idx: int, 
    pair: tuple[bytes]
) -> None:
    """
    Optimizing the merging step
    """
    new_tok = pair[0] + pair[1]
    vocab[tok_idx] = new_tok
    merges.append(pair)
        
    for prev_word in list(pair_to_words[pair]):
        if prev_word not in word_cntr:
            continue
            
        cnt = word_cntr[prev_word]
        new_word, i = [prev_word[0]], 1
        
        pairs_to_del, pairs_to_add = set(), set()
        
        while i < len(prev_word):
            if not new_word or (new_word[-1], prev_word[i]) != pair:
                new_word.append(prev_word[i])
                i += 1
                continue
            
            # Notation: ( t0 t1 t2 t3 ) -> ( t0 t1t2 t3 )
            t1 = new_word.pop()
            t2 = prev_word[i]
            t1t2 = t1 + t2
            
            # Left Neighbor Update
            if new_word:
                t0 = new_word[-1]
                pair_stats[(t0, t1)] = pair_stats.get((t0, t1), 0) - cnt
                pairs_to_del.add((t0, t1))
                
                pair_stats[(t0, t1t2)] = pair_stats.get((t0, t1t2), 0) + cnt
                pairs_to_add.add((t0, t1t2))
                
            # Right Neighbor Update
            if i < len(prev_word) - 1:
                t3 = prev_word[i+1]
                pair_stats[(t2, t3)] = pair_stats.get((t2, t3), 0) - cnt
                pairs_to_del.add((t2, t3))
                
                pair_stats[(t1t2, t3)] = pair_stats.get((t1t2, t3), 0) + cnt
                pairs_to_add.add((t1t2, t3))
            
            new_word.append(t1t2)
            i += 1
        
        new_word = tuple(new_word)
        
        word_cntr[new_word] = word_cntr.get(prev_word, 0)
        del word_cntr[prev_word]

        for pair_to_del in pairs_to_del:
            if pair_stats.get(pair_to_del, 0) > 0:
                heapq.heappush(heap, (-pair_stats[pair_to_del], ReverseLexicalPair(pair_to_del), pair_to_del))
        
        for pair_to_add in pairs_to_add:
            heapq.heappush(heap, (-pair_stats[pair_to_add], ReverseLexicalPair(pair_to_add), pair_to_add))
            
        for p in pairwise(new_word):
            pair_to_words.setdefault(p, set()).add(new_word)

    del pair_stats[pair]
    del pair_to_words[pair]


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str] | None = None, 
    n_proc: int = 8,
    vocab_path: str | None = None, 
    merges_path: str | None = None, 
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_cntr = count_words(input_path, special_tokens, n_proc)
    
    pair_stats, pair_to_words, heap = _get_initial_stats(word_cntr)

    toks = ([tok.encode("UTF-8") for tok in special_tokens] if special_tokens else []) + [bytes([i]) for i in range(256)]
    vocab = {i: tok for i, tok in enumerate(toks)}
    merges = []
    
    tok_idx = len(toks)
    pbar = tqdm(
        total=vocab_size, 
        initial=tok_idx,
        desc="Training BPE"
    )
    
    while tok_idx < vocab_size and heap:
        neg_cnt, _, pair = heapq.heappop(heap)
        
        cnt = -neg_cnt
        if pair_stats.get(pair, 0) != cnt or cnt < 1:
            continue
   
        _merge(word_cntr, pair_stats, pair_to_words, heap, vocab, merges, tok_idx, pair)
        tok_idx += 1
        pbar.update(1)
        
    pbar.close()
    
    if vocab_path:
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    if merges_path:
        with open(merges_path, 'wb') as f:
            pickle.dump(merges, f)

    return vocab, merges

