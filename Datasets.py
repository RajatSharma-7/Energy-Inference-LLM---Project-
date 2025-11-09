from datasets import load_dataset
import torch
from typing import Callable

def get_sst2(split: str = "validation"):
    """
    Load GLUE/SST-2 split.
    """
    return load_dataset("glue", "sst2")[split]

def collate_fn(tokenizer, seq_len: int = 128) -> Callable:
    """
    Fixed-length padding for fair timing and return pre-padding lengths
    for difficulty-aware analysis.
    """
    def fn(batch):
        texts = [x["sentence"] for x in batch]

        # Pre-padding lengths (difficulty proxy)
        enc_nopad = tokenizer(texts, padding=False, truncation=True, max_length=seq_len)
        lengths = [len(ids) for ids in enc_nopad["input_ids"]]

        # Fixed-length batch for timing fairness
        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=seq_len, return_tensors="pt"
        )
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc["lengths"] = torch.tensor(lengths, dtype=torch.int32)
        return {**enc, "labels": labels}
    return fn
