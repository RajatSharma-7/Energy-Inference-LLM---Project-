import os, pandas as pd, matplotlib.pyplot as plt

COLS = ["task","model","seq_len","batch_size","precision",
        "latency_ms_mean","latency_ms_std","throughput_sps",
        "seconds_total","accuracy","emissions_gco2","samples_total"]

def _need(df):
    miss = [c for c in COLS if c not in df.columns]
    if miss: raise ValueError(f"Missing cols: {miss}")

def energy_vs_latency(df, out):
    _need(df); plt.figure()
    for (p,bs), g in df.groupby(["precision","batch_size"]):
        plt.scatter(g["latency_ms_mean"], g["emissions_gco2"], label=f"{p}, bs={bs}")
    plt.xlabel("Latency (ms/sample)"); plt.ylabel("gCO₂e (run proxy)")
    plt.title("Energy vs Latency"); plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def accuracy_vs_energy(df, out):
    _need(df); plt.figure()
    for (p,bs), g in df.groupby(["precision","batch_size"]):
        plt.scatter(g["emissions_gco2"], g["accuracy"], label=f"{p}, bs={bs}")
    plt.xlabel("gCO₂e (run proxy)"); plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Energy (Pareto)"); plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def save_all(csv_path="results/runs.csv", out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    energy_vs_latency(df, os.path.join(out_dir, "energy_vs_latency.png"))
    accuracy_vs_energy(df, os.path.join(out_dir, "accuracy_vs_energy.png"))
    print("[plots] saved to", out_dir)
