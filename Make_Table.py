import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("results/runs.csv")

def mean_ci(series):
    """Return mean and 95% CI half-width for a pandas Series"""
    n = len(series)
    mean = series.mean()
    if n > 1:
        std = series.std(ddof=1)
        ci = 1.96 * (std / np.sqrt(n))
    else:
        ci = 0.0
    return mean, ci

# Aggregate manually
rows = []
for (precision, batch), group in df.groupby(["precision", "batch_size"]):
    lat_m, lat_ci = mean_ci(group["latency_ms_mean"])
    tput_m, tput_ci = mean_ci(group["throughput_sps"])
    acc_m, acc_ci = mean_ci(group["accuracy"])
    gco2_m, gco2_ci = mean_ci(group["emissions_gco2"])
    samples_mean = group["samples_total"].mean()

    # normalized emissions
    gco2_per_1000 = (gco2_m / samples_mean) * 1000.0 if samples_mean > 0 else 0.0

    rows.append({
        "Precision": precision,
        "Batch": batch,
        "Latency (ms/sample)": f"{lat_m:.2f} ± {lat_ci:.2f}",
        "Throughput (samples/s)": f"{tput_m:.2f} ± {tput_ci:.2f}",
        "Accuracy (%)": f"{acc_m:.2f} ± {acc_ci:.2f}",
        "gCO₂e (run proxy)": f"{gco2_m:.4f} ± {gco2_ci:.4f}",
        "gCO₂e / 1000 samples": f"{gco2_per_1000:.4f}"
    })

summary = pd.DataFrame(rows)

# Print Markdown for paper
print(summary.to_markdown(index=False))

# Save to CSV too
summary.to_csv("results/summary_main_table.csv", index=False)
print("\n[saved] results/summary_main_table.csv")
