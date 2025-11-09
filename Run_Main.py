import argparse, os, pandas as pd

from Build import load_model_tokenizer
from Datasets import get_sst2, collate_fn
from Eval_Task import run_eval
from Precision import fp32_strict, fp32_tf32, amp_fp16, amp_bf16
from Plotting import save_all

PREC = {
  "fp32_strict": fp32_strict,
  "fp32_tf32":   fp32_tf32,
  "fp16":        amp_fp16,
  "bf16":        amp_bf16
}

def append_csv(path, df_new):
    header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", index=False, header=header)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,8")
    ap.add_argument("--precisions", default="fp32_strict,fp32_tf32,fp16,bf16")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=150)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    model, tok = load_model_tokenizer(args.model)
    ds = get_sst2("validation")
    coll = collate_fn(tok, args.seq_len)

    for bs in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        for p in [x.strip() for x in args.precisions.split(",") if x.strip()]:
            for r in range(1, args.repeats+1):
                res = run_eval(
                    model, tok, ds, coll,
                    batch_size=bs,
                    precision_ctx=PREC[p],
                    warmup=30,
                    max_batches=args.max_batches
                )
                # Per-run aggregate row
                row = {
                    "task": args.task, "model": args.model,
                    "seq_len": args.seq_len, "batch_size": bs,
                    "precision": p, "repeat_id": r,
                    "latency_ms_mean": res["latency_ms_mean"],
                    "latency_ms_std": res["latency_ms_std"],
                    "throughput_sps": res["throughput_sps"],
                    "seconds_total": res["seconds_total"],
                    "accuracy": res["accuracy"],
                    "emissions_gco2": res["emissions_gco2"],
                    "samples_total": res["samples_total"]
                }
                append_csv("results/runs.csv", pd.DataFrame([row]))

                # Per-sample rows for difficulty analysis
                ps = pd.DataFrame(res["per_sample"])
                ps["precision"]  = p
                ps["batch_size"] = bs
                ps["seq_len"]    = args.seq_len
                ps["repeat_id"]  = r
                append_csv("results/per_sample.csv", ps)

                print(f"[DONE] precision={p} bs={bs} repeat={r} "
                      f"latency_ms={row['latency_ms_mean']:.2f} "
                      f"acc={row['accuracy']:.2f} gCO2={row['emissions_gco2']:.6f}")

    # Make the two main plots
    save_all("results/runs.csv", "plots")




'''python run_main.py --seq-len 128 --batch-sizes 1 \
  --precisions fp32_strict,fp16 --repeats 1 --max-batches 20'''    # --> This is Example for testing the code on given parameters

# Checks --> dataset load, model load, CSV writing, plots saving.