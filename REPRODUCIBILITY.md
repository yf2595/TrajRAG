# Reproducibility index (ICDM ADS 2026)

This repository supports the **controlled benchmark** in the paper: run TrajRAG, run the reported baselines, and compare mean Haversine error (MHE) to published tables. Full setup: [README.md](README.md).

**Out of scope here:** paper Section VI (real-world MLAT case studies).

## Table I — baseline → code

| Paper method | Repository |
|--------------|------------|
| **TrajRAG** | [methods/RAG.ipynb](methods/RAG.ipynb) |
| Kalman | [methods/KALMAN.ipynb](methods/KALMAN.ipynb) |
| ARIMA | [methods/ARIMA.ipynb](methods/ARIMA.ipynb) |
| LSTM | [methods/LSTM.ipynb](methods/LSTM.ipynb) |
| BiLSTM | [methods/LSTM.ipynb](methods/LSTM.ipynb) (bidirectional block) |
| Mean-Retrieval | Table V ablation *w/o LLM (mean retrieval)* — same iterative + kinematic shell, no LLM proposal |
| Numeric-kNN | [iTransformer-DLinear-PatchTST/knn_baseline.py](iTransformer-DLinear-PatchTST/knn_baseline.py) |
| DLinear, PatchTST, iTransformer | [iTransformer-DLinear-PatchTST/run_all.py](iTransformer-DLinear-PatchTST/run_all.py) |

## Other paper artifacts

| Paper | Entry point |
|-------|-------------|
| Fig. 1 | [methods/RAG.ipynb](methods/RAG.ipynb) |
| Fig. 2 | [methods/LSTM.ipynb](methods/LSTM.ipynb), [run_convergence.py](iTransformer-DLinear-PatchTST/run_convergence.py) |
| Table III | [experiments/test_multiple_LLM/](experiments/test_multiple_LLM/) |
| Table IV | [data/BCN-MUC/RAG_for_new_trajectory.ipynb](data/BCN-MUC/RAG_for_new_trajectory.ipynb) |
| Table V | [experiments/with_without_*](experiments/) |
| Sec. V-C (K) | [plots/plot_k_neighbors_vs_error.ipynb](plots/plot_k_neighbors_vs_error.ipynb) |

## Published snapshots (`results/paper/`)

| File | Table |
|------|-------|
| [table1_baselines.json](results/paper/table1_baselines.json) | I |
| [table3_llm_sensitivity.json](results/paper/table3_llm_sensitivity.json) | III |
| [table4_route_generalization.json](results/paper/table4_route_generalization.json) | IV |
| [table5_ablations.json](results/paper/table5_ablations.json) | V |

## Quick verification (no full data / API)

```bash
pip install -r requirements.txt
cd iTransformer-DLinear-PatchTST
git clone --depth 1 https://github.com/thuml/Time-Series-Library.git TSLib
python run_baseline.py --model iTransformer --epochs 2
python knn_baseline.py --runs 3
python compile_results.py --compare-paper
```
