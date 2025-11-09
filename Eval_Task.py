import time
from typing import Callable, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from Measure import TimerEnergy

def run_eval(
    model,
    tokenizer,
    dataset,
    collate: Callable,
    batch_size: int,
    precision_ctx: Callable,
    warmup: int = 30,
    max_batches: int = 200,
) -> Dict[str, Any]:
    """
    One evaluation run for a given precision & batch size.
    Returns aggregate metrics and per-sample records.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
    )

    # Warmup (stabilize kernels/clocks)
    it = iter(loader)
    for _ in range(warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        with torch.no_grad(), precision_ctx():
            inputs = {
                k: v.cuda(non_blocking=True)
                for k, v in batch.items()
                if k not in ("labels", "lengths")
            }
            _ = model(**inputs)

    # Timed loop
    latencies: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    seen_batches = 0

    with TimerEnergy() as te:
        for batch in loader:
            t0 = time.perf_counter()
            with torch.no_grad(), precision_ctx():
                inputs = {
                    k: v.cuda(non_blocking=True)
                    for k, v in batch.items()
                    if k not in ("labels", "lengths")
                }
                logits = model(**inputs).logits
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            bsz = batch["labels"].size(0)
            latency_s_per_sample = (time.perf_counter() - t0) / bsz
            latencies.append(latency_s_per_sample)

            probs = torch.softmax(logits, dim=-1).cpu()
            preds = probs.argmax(-1)
            labels = batch["labels"]
            lengths = batch["lengths"]

            # Aggregates
            correct += (preds == labels).sum().item()
            total += bsz

            # Per-sample difficulty-aware record
            for i in range(bsz):
                per_sample.append({
                    "pred": int(preds[i]),
                    "label": int(labels[i]),
                    "max_prob": float(probs[i].max()),
                    "length_tokens": int(lengths[i]),
                    "latency_s_per_sample": float(latency_s_per_sample)
                })

            seen_batches += 1
            if seen_batches >= max_batches:
                break

    acc = 100.0 * correct / total
    ms = np.array(latencies) * 1000.0

    return {
        "latency_ms_mean": float(ms.mean()),
        "latency_ms_std": float(ms.std(ddof=1) if len(ms) > 1 else 0.0),
        "throughput_sps": float(1000.0 / ms.mean()),
        "seconds_total": te.seconds,
        "accuracy": acc,
        "emissions_gco2": te.emissions_gco2,
        "samples_total": int(total),
        "per_sample": per_sample
    }
