# TrajRAG: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories

**IEEE International Conference on Data Mining (ICDM) 2026 — ADS Track**

This repository accompanies the paper:

> **TrajRAG: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories**

TrajRAG reconstructs aircraft horizontal trajectories after GNSS interference is detected. The framework discards untrusted GNSS-derived horizontal observations, retrieves phase-consistent historical flight behavior, conditions an LLM on structured ADS-B and weather states, and validates each generated position with deterministic kinematic constraints.

The repository contains the TrajRAG implementation, classical and learning-based baselines, ablation experiments, model-sensitivity experiments, data-efficiency analysis, mixed-route evaluation, and released artifacts used in the paper.

---

## Citation

```bibtex
@inproceedings{felendler2026trajrag,
  author    = {Yuval Felendler and Ruben Sasson and Yuval Elovici and Asaf Shabtai},
  title     = {{TrajRAG}: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories},
  booktitle = {2026 IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

---

## Repository layout

```text
TrajRAG/
├── TrajRAG_ADS_ICDM2026.pdf
├── trajrag/
│   └── config.py
├── methods/
├── experiments/
├── iTransformer-DLinear-PatchTST/
├── data/
├── results/paper/
├── docs/figures/
├── plots/
├── REPRODUCIBILITY.md
├── requirements.txt
├── .env.example
└── fligh_radar_api.ipynb
```

Main components:

| Component | Location | Description |
| --- | --- | --- |
| TrajRAG configuration | [`trajrag/config.py`](trajrag/config.py) | Paper hyperparameters and reconstruction settings |
| TrajRAG and classical baselines | [`methods/`](methods/) | RAG, Kalman, ARIMA, LSTM, and BiLSTM experiments |
| Modern forecasting baselines | [`iTransformer-DLinear-PatchTST/`](iTransformer-DLinear-PatchTST/) | DLinear, PatchTST, iTransformer, and Numeric-kNN |
| Ablations and model sweeps | [`experiments/`](experiments/) | Component ablations and LLM-backbone experiments |
| Data and experiment artifacts | [`data/`](data/) | Released samples, splits, embeddings, and result files |
| Published aggregates | [`results/paper/`](results/paper/) | Machine-readable paper results |
| Figure generation | [`plots/`](plots/) | Data-efficiency and sensitivity plots |
| Reproducibility index | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Paper item → repository artifact mapping |
| Data collection | [`fligh_radar_api.ipynb`](fligh_radar_api.ipynb) | FlightRadar24 + Meteomatics collection workflow |

---

## Environment

Python **3.10** is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For the TSLib baselines:

```bash
cd iTransformer-DLinear-PatchTST
git clone --depth 1 https://github.com/thuml/Time-Series-Library.git TSLib
```

Copy [`.env.example`](.env.example) to `.env` and provide the credentials required by the experiments being run.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | `gpt-5.1` generation and `text-embedding-3-large` embeddings |
| `OPENAI_API_KEY_2` | Alternate OpenAI key used by some notebooks |
| `FLIGHT_RADAR_API_KEY` | FlightRadar24 data collection |
| `METEOMATICS_USERNAME` | Meteomatics authentication |
| `METEOMATICS_PASSWORD` | Meteomatics authentication |

Do not commit `.env`, API credentials, or provider-restricted raw data.

---

## Data

### Primary controlled benchmark

The paper evaluates TrajRAG on **1,544 real commercial flights** across two European corridors.

| Corpus | Flights | Route |
| --- | ---: | --- |
| Primary evaluation | 931 | CDG → FCO |
| Mixed-route evaluation | 613 | BCN → MUC |

The CDG→FCO corpus is split chronologically:

- **831 flights** from April 19 to May 24, 2025 for training / validation and retrieval memory.
- **100 flights** from May 25 to June 1, 2025 for held-out testing.

Primary repository paths include:

| Path | Role |
| --- | --- |
| `data/CDG-FCO/flight_data_with_minutes_since_start.json` | Primary trajectory data |
| `data/CDG-FCO/RESULTS/test_sample.json` | Held-out test-flight IDs |
| `data/MULTI_ROUTE/EMBEDDINGS/` | Phase-specific FAISS indices |
| `data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json` | TrajRAG per-flight reconstruction errors |
| `data/BCN-MUC/` | Barcelona→Munich mixed-route evaluation |
| `data/{MRS-MUC,NTE-MUC,BCN-MUC,BCN-CDG,CDG-FCO,BOD-VCE}/` | Multi-route forecasting inputs |

Where provider terms prevent redistribution of complete raw data, [`fligh_radar_api.ipynb`](fligh_radar_api.ipynb) documents the collection workflow using FlightRadar24 and Meteomatics.

---

## Controlled GNSS-blackout protocol

The paper evaluates post-detection reconstruction rather than denoising.

Once GNSS interference is detected:

1. GNSS-derived horizontal fields are treated as untrusted.
2. Post-detection latitude, longitude, ground speed, and heading are not supplied to the reconstruction loop.
3. Barometric altitude and timestamps remain available.
4. Reconstruction starts from the last verified state.
5. TrajRAG iteratively retrieves phase-consistent historical segments, proposes the next latitude/longitude, and applies deterministic kinematic validation.
6. Reconstruction continues until trusted reporting resumes or the flight terminates.

The controlled evaluation uses randomized spoofing onset times and blackout durations of **5–15 time steps**, corresponding to approximately **10–30 minutes** at the dataset's approximately 120-second sampling interval.

Some development notebooks also contain fixed-midpoint blackout settings such as `SPOOF_INDEX = N // 2`. Those runs are diagnostic configurations; the paper protocol is the randomized corruption setting described above.

---

## TrajRAG configuration

Paper defaults are defined in [`trajrag/config.py`](trajrag/config.py).

| Parameter | Value |
| --- | --- |
| Generation model | `gpt-5.1` |
| Generation temperature | `0` |
| Text-state summarization model | `gpt-5.1` |
| Summarization temperature | `0.3` |
| Embedding model | `text-embedding-3-large` |
| Retrieval neighbors | `K = 5` |
| Retrieval similarity | Cosine similarity via L2-normalized FAISS inner-product search |
| Phase memories | Takeoff, cruise, landing |
| Kinematic tolerance | `α = 1.2` |
| Nominal sampling interval | `Δt = 120 s` |
| Feedback loop | Enabled |

The validator constrains each proposed displacement according to:

\[
d_{\max} = \alpha v_t \Delta t
\]

where \(v_t\) is the previous accepted ground speed.

---

## Baselines

The paper compares TrajRAG with classical, recurrent, modern time-series, and retrieval-only baselines.

| Method | Configuration | Entry point |
| --- | --- | --- |
| Kalman | Phase-wise classical filtering | [`methods/KALMAN.ipynb`](methods/KALMAN.ipynb) |
| ARIMA | Statistical per-phase forecasting | [`methods/ARIMA.ipynb`](methods/ARIMA.ipynb) |
| LSTM / BiLSTM | hidden=128, 6 layers, Adam lr=1e-2, batch=64, 100 epochs | [`methods/LSTM.ipynb`](methods/LSTM.ipynb) |
| Mean-Retrieval | Top-k neighbor future-state averaging without an LLM | Ablation notebooks |
| Numeric-kNN | k=5 using standardized numeric prefix features | [`iTransformer-DLinear-PatchTST/knn_baseline.py`](iTransformer-DLinear-PatchTST/knn_baseline.py) |
| DLinear | TSLib forecasting baseline | [`iTransformer-DLinear-PatchTST/run_all.py`](iTransformer-DLinear-PatchTST/run_all.py) |
| PatchTST | TSLib forecasting baseline | [`iTransformer-DLinear-PatchTST/run_all.py`](iTransformer-DLinear-PatchTST/run_all.py) |
| iTransformer | TSLib forecasting baseline | [`iTransformer-DLinear-PatchTST/run_all.py`](iTransformer-DLinear-PatchTST/run_all.py) |

---

## Metric

Reconstruction accuracy is measured with **Mean Haversine Error (MHE)** in kilometers:

\[
\mathrm{MHE} =
\frac{1}{|T|}
\sum_{t \in T}
d_{\mathrm{hav}}(\hat{x}_t, x_t)
\]

where \(T\) is the set of reconstructed indices, \(\hat{x}_t\) is the reconstructed position, and \(x_t\) is the uncorrupted reference position.

---

## Table I — comparison with baselines

Mean Haversine error in km, reported as mean ± standard deviation.

| Method | Takeoff | Cruise | Landing |
| --- | ---: | ---: | ---: |
| Kalman | 60.95 ± 30.45 | 39.06 ± 25.10 | 112.61 ± 31.78 |
| ARIMA | 112.34 ± 14.00 | 37.69 ± 37.57 | 73.82 ± 22.55 |
| LSTM | 33.78 ± 23.11 | 40.66 ± 22.72 | 18.62 ± 14.33 |
| BiLSTM | 35.79 ± 31.58 | 46.83 ± 27.71 | 11.17 ± 4.91 |
| Mean-Retrieval | 48.94 ± 37.77 | 30.80 ± 21.64 | 8.52 ± 6.42 |
| Numeric-kNN | 66.19 ± 29.34 | 32.49 ± 17.12 | 16.92 ± 10.20 |
| DLinear | 43.99 ± 13.18 | 21.01 ± 7.03 | 35.47 ± 11.06 |
| PatchTST | 55.77 ± 10.80 | 23.04 ± 8.90 | 38.17 ± 9.28 |
| iTransformer | 47.89 ± 8.08 | 19.79 ± 5.40 | 37.58 ± 7.35 |
| **TrajRAG** | **20.78 ± 4.60** | **9.84 ± 4.40** | **2.46 ± 2.33** |

Machine-readable paper values:

[`results/paper/table1_baselines.json`](results/paper/table1_baselines.json)

---

## Table II — reconstruction latency and generation cost

| Phase | Text construction (s) | Retrieval (s) | Generation (s) | Total (s) | Cost ($) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Takeoff | 3.2 | 0.53 | 1.5 | 5.23 | 0.014 |
| Cruise | 4.0 | 0.51 | 2.3 | 6.81 | 0.018 |
| Landing | 3.4 | 0.49 | 1.7 | 5.59 | 0.016 |

Text construction is incurred once at blackout onset. The repeated retrieval + generation loop remains below the approximately 120-second ADS-B sampling interval.

---

## Table III — LLM backbone sensitivity

| Model | Takeoff μ | Takeoff σ | Cruise μ | Cruise σ | Landing μ | Landing σ | Input cost / 1M tokens ($) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4.1 | 25.14 | 5.01 | 7.13 | 3.46 | 2.98 | 2.14 | 2.00 |
| gpt-4o | 22.47 | 5.06 | 10.88 | 7.55 | 13.85 | 12.41 | 2.50 |
| gpt-4o-mini | 86.80 | 39.10 | 47.70 | 23.99 | 19.65 | 15.47 | 0.15 |
| **gpt-5.1** | **20.78** | **4.60** | 9.84 | 4.40 | **2.46** | **2.33** | 1.25 |
| gpt-5.2 | 20.82 | 4.86 | **7.12** | **3.32** | 2.62 | 2.53 | 1.75 |
| Llama 3.3 7B | 52.89 | 13.22 | 52.16 | 13.04 | 13.43 | 3.36 | 0.27 |
| Mistral 7B | 59.79 | 14.95 | 58.96 | 14.74 | 15.18 | 3.80 | 0.10 |
| Qwen3-Max-Thinking | 41.39 | 10.35 | 40.82 | 10.21 | 10.51 | 2.63 | 0.20 |

Entry point:

[`experiments/test_multiple_LLM/test_multiple_LLM.ipynb`](experiments/test_multiple_LLM/test_multiple_LLM.ipynb)

---

## Table IV — mixed-route memory robustness

The paper compares route-specific memory (S1) with a mixed-route memory containing trajectories from multiple corridors (S2).

Values report the change in MHE from S1 to S2.

| Route | Takeoff | Cruise | Landing |
| --- | ---: | ---: | ---: |
| Paris→Rome | +0.09 | -0.11 | +0.71 |
| Barcelona→Munich | +0.74 | -0.02 | -0.06 |

Entry point:

[`data/BCN-MUC/RAG_for_new_trajectory.ipynb`](data/BCN-MUC/RAG_for_new_trajectory.ipynb)

---

## Table V — ablation studies

The ablation study removes one component at a time while keeping the remaining pipeline fixed.

The evaluated configurations include:

- w/o retrieval
- w/o LLM (mean retrieval)
- w/o kinematic validation
- w/o feedback loop
- w/o textual representation
- w/o weather data
- w/o barometric altitude
- full TrajRAG

Entry points:

```text
experiments/with_without_*
experiments/generate_full_table.ipynb
```

The paper reports paired two-sided Wilcoxon signed-rank tests over per-flight reconstruction errors.

---

## Figure 2 — data efficiency

The data-efficiency experiment varies the amount of available historical training/reference data and compares TrajRAG with LSTM and iTransformer.

Entry points:

```text
methods/LSTM.ipynb
run_convergence.py
data/EXPERIMENTS/MSE_CONVERGENCE_SPEED/
```

The corresponding plotting code is under [`plots/`](plots/).

---

## Real-world MLAT-supported case studies

The controlled benchmark is complemented by four real GNSS-interference episodes for which FlightRadar24 reported **Terrestrial MLAT** positions.

Only positions explicitly identified as Terrestrial MLAT are used as the independent consistency reference during the interference window.

| Flight | Route | Region / onset | Window | Memory | Mean MLAT deviation (km) | Endpoint deviation (km) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A6-AQF | AMM → AUH | Saudi Arabian interior | 103 min | 200 | 6.96 | 1.97 |
| TG941 | MXP → BKK | Black Sea / Northern Turkey | 46 min | 200 | 7.63 | 4.78 |
| CK602 | BUD → CKG | Baltic region | 71 min | 200 | 7.95 | 5.01 |
| QR8213 | DOH → MAD | Sinai–Red Sea / Gulf of Aqaba | 53 min | 200 | 6.95 | 6.36 |

The case-study artifacts and paper-item mapping are documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Reproduction workflow

### 1 — Smoke test

```bash
cd iTransformer-DLinear-PatchTST
python run_baseline.py --model iTransformer --epochs 2
python knn_baseline.py --runs 3
```

### 2 — Prepare data

Use:

[`fligh_radar_api.ipynb`](fligh_radar_api.ipynb)

to collect and preprocess the available flight and weather data into the route-specific format expected by the experiments.

### 3 — Run TrajRAG

Open:

[`methods/RAG.ipynb`](methods/RAG.ipynb)

The notebook performs:

1. textual-state construction,
2. phase-specific embedding and FAISS indexing,
3. iterative retrieval-conditioned reconstruction,
4. kinematic validation,
5. per-flight MHE evaluation.

Primary output:

```text
data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json
```

### 4 — Run classical baselines

| Method | Entry point |
| --- | --- |
| Kalman | [`methods/KALMAN.ipynb`](methods/KALMAN.ipynb) |
| ARIMA | [`methods/ARIMA.ipynb`](methods/ARIMA.ipynb) |
| LSTM / BiLSTM | [`methods/LSTM.ipynb`](methods/LSTM.ipynb) |

### 5 — Run modern forecasting baselines

```bash
cd iTransformer-DLinear-PatchTST
python adapt_data.py
python run_all.py --itransformer --bonus
python compile_results.py --compare-paper
```

Detailed commands:

[`iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md`](iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md)

### 6 — Run extended analyses

| Paper item | Entry point |
| --- | --- |
| Fig. 2 data efficiency | `methods/LSTM.ipynb` + `run_convergence.py` |
| Table III LLM sweep | `experiments/test_multiple_LLM/test_multiple_LLM.ipynb` |
| Table IV mixed-route memory | `data/BCN-MUC/RAG_for_new_trajectory.ipynb` |
| Table V ablations | `experiments/with_without_*` + `experiments/generate_full_table.ipynb` |
| K-neighbor sensitivity | `plots/plot_k_neighbors_vs_error.ipynb` |

---

## Verify against the paper

Windows:

```bash
type results\paper\table1_baselines.json
```

Linux / macOS:

```bash
cat results/paper/table1_baselines.json
```

For the TSLib / Numeric-kNN comparison:

```bash
cd iTransformer-DLinear-PatchTST
python compile_results.py --compare-paper
```

The paper-to-artifact index is maintained in:

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

---

## Data and use conditions

Flight and weather data remain subject to the terms of the services from which they were obtained. The repository does not relicense provider-restricted raw data.

The public artifact is intended for research reproducibility. API credentials, `.env` files, private data, and provider-restricted datasets are not part of the repository.
