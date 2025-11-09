import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

ps_path = "results/per_sample.csv"
runs_path = "results/runs.csv"
assert os.path.exists(ps_path) and os.path.exists(runs_path), "Run experiments first."

ps = pd.read_csv(ps_path)
runs = pd.read_csv(runs_path)

# Quintile bins for difficulty
ps["len_bin"]  = pd.qcut(ps["length_tokens"], q=5, labels=[1,2,3,4,5])
ps["conf_bin"] = pd.qcut(ps["max_prob"], q=5, labels=[1,2,3,4,5])

def agg(df, by):
    return df.groupby(by).apply(lambda x: pd.Series({
        "n": len(x),
        "acc": 100.0 * (x["pred"]==x["label"]).mean(),
        "latency_ms": 1000.0 * x["latency_s_per_sample"].mean()
    })).reset_index()

os.makedirs("plots", exist_ok=True)
for facet, label in [("len_bin","Length Quintile (1=short,5=long)"),
                     ("conf_bin","Confidence Quintile (1=low,5=high)")]:
    for bs in sorted(ps["batch_size"].unique()):
        g = agg(ps[ps["batch_size"]==bs], ["precision", facet])
        # Merge mean emissions per (precision, batch) for context
        em = runs[runs["batch_size"]==bs].groupby(["precision"], as_index=False)["emissions_gco2"].mean()
        g = g.merge(em, on="precision", how="left")

        # Accuracy vs Energy (per bin)
        plt.figure()
        for p, grp in g.groupby("precision"):
            plt.plot(grp["emissions_gco2"], grp["acc"], marker="o", label=p)
        plt.xlabel("gCO₂e (run proxy)"); plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy vs Energy by {label} (batch={bs})")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/pareto_{facet}_bs{bs}.png"); plt.close()

        # Accuracy vs bin index
        plt.figure()
        for p, grp in g.groupby("precision"):
            plt.plot(grp[facet].astype(int), grp["acc"], marker="o", label=p)
        plt.xlabel(label); plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy by {label} (batch={bs})")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/acc_by_{facet}_bs{bs}.png"); plt.close()

print("[difficulty] plots saved in plots/")
