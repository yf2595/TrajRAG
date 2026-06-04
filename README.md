# TrajRAG: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories

**ICDM ADS 2026** — Reproducibility artifact for [TrajRAG_ADS_ICDM2026.pdf](TrajRAG_ADS_ICDM2026.pdf).

## Scope of this repository

This repo is for **running TrajRAG** and **evaluating it against the baselines reported in the paper** under the controlled GNSS-blackout benchmark (simulated post-detection corruption, ground-truth ADS-B for scoring).


TrajRAG reconstructs horizontal positions after spoof detection by retrieving phase-consistent historical flights, conditioning **gpt-5.1** on textual ADS-B/weather states, and validating each LLM proposal with a kinematic envelope ([trajrag/config.py](trajrag/config.py)).

---

## Citation

```bibtex
@inproceedings{trajrag2026icdm,
  title={TrajRAG: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories},
  booktitle={IEEE International Conference on Data Mining (ICDM), ADS Track},
  year={2026},
  note={ADS 2026}
}
```

---

## Repository layout

```
TrajRAG/
├── TrajRAG_ADS_ICDM2026.pdf
├── trajrag/config.py              # Paper hyperparameters (Sec. IV-G)
├── methods/                       # TrajRAG + classical baselines
├── iTransformer-DLinear-PatchTST/ # TSLib models + Numeric-kNN
├── experiments/                   # Ablations, LLM sweep, sample-efficiency
├── data/                          # Sample JSON + convergence batch IDs
├── results/paper/                 # Published Table I/III–V aggregates
├── docs/figures/                  # Fig. 1, 2, K-sensitivity
├── plots/                         # Regenerate figures
└── fligh_radar_api.ipynb          # Collect trajectories (FR24 + Meteomatics)
```

---

## Environment


| Requirement | Notes                                             |
| ----------- | ------------------------------------------------- |
| Python      | **3.10** recommended                              |
| GPU         | Optional (TSLib); CPU OK for TrajRAG (OpenAI API) |
| TSLib       | Clone into `iTransformer-DLinear-PatchTST/TSLib`  |


```bash
python -m venv .venv
# .venv\Scripts\activate   (Windows)  |  source .venv/bin/activate  (Unix)
pip install -r requirements.txt

cd iTransformer-DLinear-PatchTST
git clone --depth 1 https://github.com/thuml/Time-Series-Library.git TSLib
```

Copy [.env.example](.env.example) to `.env`:


| Variable                                        | Purpose                             |
| ----------------------------------------------- | ----------------------------------- |
| `OPENAI_API_KEY`                                | Embeddings + **gpt-5.1** generation |
| `OPENAI_API_KEY_2`                              | Alias in some notebooks             |
| `FLIGHT_RADAR_API_KEY`                          | Data collection                     |
| `METEOMATICS_USERNAME` / `METEOMATICS_PASSWORD` | Weather at collection time          |


---

## Data (paper Sec. IV-A)


| Corpus                  | Flights                        | Route               | In this repo  |
| ----------------------- | ------------------------------ | ------------------- | ------------- |
| Primary evaluation      | 931 (831 train + **100 test**) | CDG → FCO           | Sample JSON   |
| Route generalization    | 613                            | BCN → MUC           | Sample JSON   |
| TSLib / KNN multi-route | 6 corridors                    | See `adapt_data.py` | 2 / 6 folders |


**Paths for a full rerun:**


| Path                                                                                               | Role                                 |
| -------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `data/CDG-FCO/flight_data_with_minutes_since_start.json`                                           | Primary trajectories                 |
| `data/CDG-FCO/RESULTS/test_sample.json`                                                            | 100 held-out test IDs                |
| `data/MULTI_ROUTE/EMBEDDINGS/`                                                                     | Phase FAISS indices (built offline)  |
| `data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json`                                                 | TrajRAG per-flight MHE (eval output) |
| `data/BCN-MUC/...`                                                                                 | Table IV generalization              |
| `data/{MRS-MUC,NTE-MUC,BCN-MUC,BCN-CDG,CDG-FCO,BOD-VCE}/flight_data_with_minutes_since_start.json` | Six-route TSLib CSV                  |


Collect with [fligh_radar_api.ipynb](fligh_radar_api.ipynb). Compare to published aggregates in [results/paper/](results/paper/) without re-running APIs.

---

## Evaluation protocol (aligned with paper)

Matches **Sec. IV** and the notebook implementation used for Tables I–V.


| Item                       | Paper / repo setting                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| Setting                    | Post-detection: discard untrusted **horizontal** GNSS fields; keep **barometric altitude** |
| Phases                     | Takeoff, cruise, landing (barometric-altitude segmentation)                                |
| Blackout per phase segment | First **N/2** steps observed; remaining steps reconstructed (`SPOOF_INDEX = N // 2`)       |
| Metric                     | **MHE** — mean great-circle (Haversine) error in **km**, per phase                         |
| Primary split              | Chronological **831** train + **100** test (CDG–FCO, Apr 19–Jun 1, 2025)                   |
| Parametric baselines       | Chronological **83% / 17%** flight split (LSTM, TSLib, KNN harness)                        |
| Sampling                   | ~**120 s** between ADS-B states                                                            |
| Retrieval                  | **K = 5**, L2-normalized embeddings, cosine via FAISS, **per-phase** index                 |


---

## Hyperparameters (paper Sec. IV-G)

Defined in [trajrag/config.py](trajrag/config.py).

### TrajRAG


| Parameter       | Value                                                   |
| --------------- | ------------------------------------------------------- |
| Generation LLM  | `gpt-5.1`, temperature **0**                            |
| Summary LLM     | `gpt-5.1`, temperature **0.3** (offline textual states) |
| Embedding       | `text-embedding-3-large`                                |
| Retrieval **K** | **5**                                                   |
| Kinematic **α** | **1.2** (Eq. 1, Δt = 120 s)                             |
| Feedback loop   | On (off in ablation notebooks)                          |


### Table I baselines


| Method                            | Config                                                       | Entry point                                                      |
| --------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| Kalman                            | Phase-wise filter, same blackout                             | [methods/KALMAN.ipynb](methods/KALMAN.ipynb)                     |
| ARIMA                             | Statistical per phase                                        | [methods/ARIMA.ipynb](methods/ARIMA.ipynb)                       |
| LSTM / BiLSTM                     | hidden=128, layers=6, Adam lr=1e-2, batch=64, 100 epochs     | [methods/LSTM.ipynb](methods/LSTM.ipynb)                         |
| **Mean-Retrieval**                | Top-k neighbor **averaging** (no LLM); see Table V *w/o LLM* | Ablation notebooks                                               |
| **Numeric-kNN**                   | k=5, numeric prefix features, variable blackout              | [knn_baseline.py](iTransformer-DLinear-PatchTST/knn_baseline.py) |
| DLinear / PatchTST / iTransformer | seq/pred **5**, epochs **30**, seeds 0–2, lr=1e-3, batch=64  | [run_all.py](iTransformer-DLinear-PatchTST/run_all.py)           |


---

## Published Table I (TrajRAG vs. baselines, km)

Values from the paper; local reruns should target these after full data + **gpt-5.1** eval.


| Method       | Takeoff μ±σ   | Cruise μ±σ    | Landing μ±σ   |
| ------------ | ------------- | ------------- | ------------- |
| TrajRAG      | 20.78 ± 4.60  | 9.84 ± 4.40   | 2.46 ± 2.33   |
| iTransformer | 47.89 ± 8.08  | 19.79 ± 5.40  | 37.58 ± 7.35  |
| DLinear      | 43.99 ± 13.18 | 21.01 ± 7.03  | 35.47 ± 11.06 |
| Numeric-kNN  | 66.19 ± 29.34 | 32.49 ± 17.12 | 16.92 ± 10.20 |
| LSTM         | 33.78 ± 23.11 | 40.66 ± 22.72 | 18.62 ± 14.33 |


Full table: [results/paper/table1_baselines.json](results/paper/table1_baselines.json).

---

## How to reproduce

### 0 — Smoke test

```bash
cd iTransformer-DLinear-PatchTST
python run_baseline.py --model iTransformer --epochs 2
python knn_baseline.py --runs 3
```

### 1 — Data (if not using a private copy)

[fligh_radar_api.ipynb](fligh_radar_api.ipynb) → preprocess → phase JSON under `data/<route>/`.

### 2 — TrajRAG → Table I

[methods/RAG.ipynb](methods/RAG.ipynb):

1. Offline: textual summaries → `data/MULTI_ROUTE/EMBEDDINGS/`
2. Online: reconstruct 100 test flights → `data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json`

### 3 — Classical baselines → Table I


| Notebook                             | Output (typical)                            |
| ------------------------------------ | ------------------------------------------- |
| [KALMAN.ipynb](methods/KALMAN.ipynb) | `MEAN_HAVERSINE_KALMAN.json`                |
| [ARIMA.ipynb](methods/ARIMA.ipynb)   | phase MHE in `RESULTS/`                     |
| [LSTM.ipynb](methods/LSTM.ipynb)     | `MEAN_HAVERSINE_LSTM.json` (+ BiLSTM block) |


### 4 — TSLib + Numeric-kNN → Table I

```bash
cd iTransformer-DLinear-PatchTST
python adapt_data.py
python run_all.py --itransformer --bonus
python compile_results.py --compare-paper
```

CLI details: [RUN_EXPERIMENTS.md](iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md).

### 5 — Extended analyses (same benchmark)


| Goal                        | Command / notebook                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Fig. 2 data efficiency      | [LSTM.ipynb](methods/LSTM.ipynb) + `run_convergence.py`; batches in `data/EXPERIMENTS/MSE_CONVERGENCE_SPEED/` |
| Table III LLM sweep         | [test_multiple_LLM.ipynb](experiments/test_multiple_LLM/test_multiple_LLM.ipynb)                              |
| Table IV mixed-route memory | [RAG_for_new_trajectory.ipynb](data/BCN-MUC/RAG_for_new_trajectory.ipynb)                                     |
| Table V ablations           | `experiments/with_without_*` → [generate_full_table.ipynb](experiments/generate_full_table.ipynb)             |
| K sensitivity               | [plot_k_neighbors_vs_error.ipynb](plots/plot_k_neighbors_vs_error.ipynb)                                      |


---

## Verify against the paper

```bash
# Published aggregates
type results\paper\table1_baselines.json    # Windows
# cat results/paper/table1_baselines.json   # Unix

cd iTransformer-DLinear-PatchTST
python compile_results.py --compare-paper
```

`--compare-paper` prints per-method Δ vs. [table1_baselines.json](results/paper/table1_baselines.json). TSLib/KNN reruns in this repo already match Table I when six-route data is complete; TrajRAG requires `MEAN_HAVERSINE_RAG.json` from step 2.

---

## Protocol notes (avoid mismatches)

1. **TSLib `pred_len=5` vs. TrajRAG variable horizon** — Transformer baselines use a fixed 5-step window; TrajRAG and Numeric-kNN use **N − N/2** per phase. Only the latter two are directly comparable on horizon length; TSLib numbers follow the paper’s forecasting baseline setup.
2. **Primary eval uses 100 test flights (831/100 split)** — TSLib/KNN harness uses **83%/17%** chronological split over adapted multi-route CSV. Same metric (MHE), slightly different split definition as in the paper’s baseline section.
3. `**gpt-5.1`** — Table I TrajRAG numbers assume the backbone in Sec. IV-G. Older snapshots (e.g. `gpt-4.1`) will not match [table1_baselines.json](results/paper/table1_baselines.json).
4. **Mean-Retrieval** — Reported in Table I as a retrieval-only baseline; reproduced via the *w/o LLM (mean retrieval)* ablation (Table V), not a separate top-level script.

---

## Checklist

- Python 3.10 + `pip install -r requirements.txt`
- TSLib cloned
- `.env` with OpenAI key
- CDG–FCO data + `test_sample.json` (or verify via `results/paper/`)
- TrajRAG: embeddings built, eval → `MEAN_HAVERSINE_RAG.json`
- Baselines + `compile_results.py --compare-paper`

---

## More docs

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — Paper table → file index
- [results/paper/README.md](results/paper/README.md) — JSON snapshots
- [iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md](iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md) — TSLib steps

---

## License

Flight and weather data are subject to third-party API terms. Do not commit `.env` or proprietary datasets.
