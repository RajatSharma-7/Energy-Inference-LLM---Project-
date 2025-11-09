# ⚡ Energy Inference in LLMs: Analyzing Energy–Latency–Accuracy Trade-offs in Transformer Model Inference

This repository contains the complete implementation, results, and documentation for my B.Tech Minor Project **“Energy Inference in LLMs”**, focused on quantifying how different numerical precision modes and batch configurations influence the **energy consumption**, **latency**, and **accuracy** of Transformer-based language models.

---

## 📘 Project Overview

Large Language Models (LLMs) such as BERT and RoBERTa achieve remarkable performance but are computationally and environmentally expensive.  
This project investigates the **Energy–Latency–Accuracy (ELA)** trade-off in Transformer inference using a controlled experimental pipeline.

The goal is to identify **Pareto-optimal configurations** that balance performance and sustainability — enabling faster and greener AI inference.

---

## 🧠 Objectives

- Analyze how **numerical precision (FP32, TF32, FP16, BF16)** affects latency, throughput, and energy.  
- Measure **energy consumption and CO₂ emissions** during inference using [CodeCarbon](https://mlco2.github.io/codecarbon/).  
- Compare batch sizes (1 vs 8) to assess the effect of parallelism on efficiency.  
- Visualize **Pareto fronts** for energy-latency trade-offs and identify optimal configurations.  
- Establish a reproducible benchmark for **Green AI research** in NLP.

---

## ⚙️ Experimental Setup

| Category | Details |
|-----------|----------|
| **Model** | RoBERTa-base (12 layers, 12 heads, ~125M params) |
| **Dataset** | SST-2 (Stanford Sentiment Treebank v2, GLUE benchmark) |
| **Frameworks** | PyTorch 2.x, Hugging Face Transformers, Datasets |
| **Energy Tracker** | CodeCarbon v2.0 |
| **Hardware** | Intel i7-12th Gen, NVIDIA RTX 3050 (4GB VRAM), 16GB RAM |
| **Precision Modes** | FP32, TF32, FP16, BF16 |
| **Batch Sizes** | 1 and 8 |
| **Metrics** | Accuracy, Latency (ms/sample), Throughput (samples/s), Energy (kWh), Emissions (gCO₂e) |

---

## 🧩 Methodology

1. **Dataset Preparation** – Tokenize and preprocess SST-2 with RoBERTa tokenizer.  
2. **Model Setup** – Load pre-trained RoBERTa-base from Hugging Face.  
3. **Precision Control** – Run inference under FP32, TF32, FP16, and BF16 using PyTorch AMP.  
4. **Batch Variation** – Compare per-sample performance at batch sizes 1 and 8.  
5. **Energy Logging** – Record real-time power draw and emissions with CodeCarbon.  
6. **Result Aggregation** – Compute mean ± SD across five runs.  
7. **Visualization** – Plot Accuracy–Energy, Energy–Latency, and Pareto efficiency fronts.

---

## 📊 Key Findings

| Metric | FP32 | FP16 | BF16 |
|---------|------|------|------|
| **Accuracy (%)** | 94.7 | 94.4 | 94.5 |
| **Latency ↓** | 25.6 → 17.5 ms | — | — |
| **Energy ↓** | 0.016 → 0.012 kWh | — | — |
| **CO₂ ↓** | 8.4 → 6.4 g | — | — |

- **FP16/BF16** achieved ~30% lower latency and ~25% lower emissions vs FP32.  
- **Batch 8** improved throughput 2–3× while reducing per-sample energy by ~20%.  
- Accuracy remained nearly constant across all precision settings.  
- Pareto analysis identified **FP16 & BF16 (Batch 8)** as optimal for sustainable inference.

---

## 📈 Visual Results

Key plots included:
- Accuracy vs Precision  
- Latency vs Throughput  
- Energy vs Precision  
- Pareto Front (Latency–Emissions)  
- Difficulty-Aware Efficiency (Sentence Length & Confidence)

All figures are available in the `/results` folder.

---

## 🧪 How to Reproduce

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/energy-inference-llms.git
cd energy-inference-llms

# 2. Create environment
python -m venv .venv
source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run inference and log energy
python src/energy_inference.py --precision fp16 --batch 8 --dataset sst2

# 5. View generated results and visualizations
open results/
