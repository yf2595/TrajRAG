# TrajRAG: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories

**IEEE ICDM 2026 — ADS Track Reproducibility Artifact**

## Scope of this repository

This repository accompanies the paper:

> **TrajRAG: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories**

It contains the code and released experimental artifacts for running TrajRAG and evaluating it against the baselines reported in the paper under the controlled post-detection GNSS-blackout benchmark.

TrajRAG reconstructs horizontal positions after spoofing detection by retrieving phase-consistent historical flights, conditioning **gpt-5.1** on textual ADS-B/weather states, and validating each LLM proposal with a deterministic kinematic envelope defined in [`trajrag/config.py`](trajrag/config.py).

The public artifact focuses on the reproducible research pipeline. Third-party data remain subject to their providers' terms, and some real-world operational case-study inputs may need to be reconstructed through the documented APIs.

---

## Citation

If you use this code, data, or evaluation artifact, please cite:

```bibtex
@inproceedings{felendler2026trajrag,
  author    = {Yuval Felendler and Ruben Sasson and Yuval Elovici and Asaf Shabtai},
  title     = {{TrajRAG}: Textual State-Driven Retrieval-Augmented Iterative Reconstruction of Spoofed Aircraft Trajectories},
  booktitle = {2026 IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

Final DOI, page numbers, and proceedings metadata can be added after IEEE publication.

---

## Repository layout

```text
TrajRAG/
├── TrajRAG_ADS_ICDM2026.pdf
├── trajrag/
│   └── config.py                    # Paper hyperparameters (Sec. IV-G)
├── methods/                         # TrajRAG + classical baselines
├── iTransformer-DLinear-PatchTST/  # TSLib models + Numeric-kNN
├── experiments/                     # Ablations, LLM sweep, sample-efficiency
├── data/                            # Released samples, splits, and experiment artifacts
├── results/paper/                   # Published aggregate results
├── docs/figures/                    # Paper figures / supporting plots
├── plots/                           # Figure-generation notebooks/scripts
├── REPRODUCIBILITY.md               # Paper item -> repository artifact map
└── fligh_radar_api.ipynb            # FR24 + Meteomatics data collection
```

---

## Environment

| Requirement | Notes |
| --- | --- |
| Python | **3.10** recommended |
| GPU | Optional for TrajRAG API runs; required/recommended for local TSLib/open-weight experiments |
| TSLib | Clone into `iTransformer-DLinear-PatchTST/TSLib` |

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

cd iTransformer-DLinear-PatchTST
git clone --depth 1 https://github.com/thuml/Time-Series-Library.git TSLib
```

Copy [`.env.example`](.env.example) to `.env` and provide only the credentials needed for the experiments you run.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Embeddings + **gpt-5.1** generation |
| `OPENAI_API_KEY_2` | Alias used by some notebooks |
| `FLIGHT_RADAR_API_KEY` | Flight-data collection |
| `METEOMATICS_USERNAME` / `METEOMATICS_PASSWORD` | Weather collection |

Do not commit `.env`, API keys, credentials, or provider-restricted datasets.

---

## Data

### Primary controlled benchmark

| Corpus | Flights | Route | Public artifact |
| --- | ---: | --- | --- |
| Primary evaluation | 931 (831 train/validation + **100 test**) | CDG → FCO | Released samples / reproducibility metadata |
| Mixed-route evaluation | 613 | BCN → MUC | Released samples / reproducibility metadata |
| TSLib / Numeric-kNN multi-route | 6 corridors | See `adapt_data.py` | Partial route folders + reconstruction instructions |

Primary data paths used by the repository include:

| Path | Role |
| --- | --- |
| `data/CDG-FCO/flight_data_with_minutes_since_start.json` | Primary trajectories |
| `data/CDG-FCO/RESULTS/test_sample.json` | 100 held-out test IDs |
| `data/MULTI_ROUTE/EMBEDDINGS/` | Phase-specific FAISS indices |
| `data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json` | TrajRAG per-flight MHE output |
| `data/BCN-MUC/...` | Mixed-route evaluation |
| `data/{MRS-MUC,NTE-MUC,BCN-MUC,BCN-CDG,CDG-FCO,BOD-VCE}/flight_data_with_minutes_since_start.json` | Six-route TSLib inputs |

Data collection / reconstruction is documented in [`fligh_radar_api.ipynb`](fligh_radar_api.ipynb). Where redistribution is restricted by provider terms, use the documented FlightRadar24 and Meteomatics queries instead of expecting a complete raw-data dump.

Published aggregate values can be checked directly under [`results/paper/`](results/paper/) without re-running paid APIs.

---

## Evaluation protocol

The controlled evaluation follows the paper's post-detection recovery setting:

| Item | Paper / repository setting |
| --- | --- |
| Trust boundary | After detection, discard untrusted **horizontal GNSS-derived fields** |
| Trusted auxiliary signal | Keep **barometric altitude** and timestamps |
| Phases | Takeoff, cruise, landing |
| Split | Chronological **831 train/validation + 100 held-out test flights** for CDG→FCO |
| Blackout | Randomized spoofing onset and **5–15 missing time steps** (~10–30 min at ~120 s sampling) |
| Metric | **MHE** — mean Haversine distance error in km |
| Retrieval | **K = 5**, L2-normalized embeddings, cosine similarity via FAISS |
| Memory | Separate **phase-aware** indices |
| Reconstruction | Iterative retrieval → LLM proposal → kinematic validation |
| Sampling | Approximately **120 s** between ADS-B states |

### Important protocol note

Some development or diagnostic notebooks may expose a fixed-midpoint blackout such as:

```python
SPOOF_INDEX = N // 2
```

That fixed-midpoint setting is useful for debugging, but it should **not** be treated as the paper's randomized corruption protocol. Paper reproduction should use the recorded randomized onset and 5–15-step blackout settings described in the final manuscript and indexed in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Hyperparameters

Paper defaults are defined in [`trajrag/config.py`](trajrag/config.py).

### TrajRAG

| Parameter | Value |
| --- | --- |
| Generation LLM | `gpt-5.1`, temperature **0** |
| Summary LLM | `gpt-5.1`, temperature **0.3** for offline textual-state construction |
| Embedding | `text-embedding-3-large` |
| Retrieval **K** | **5** |
| Kinematic **α** | **1.2** |
| Nominal **Δt** | **120 s** |
| Feedback loop | Enabled |
| Retrieval refresh | Recomputed after each accepted reconstructed state |

---

## Baselines

### Table I baselines

| Method | Configuration | Entry point |
| --- | --- | --- |
| Kalman | Phase-wise filter under the same blackout setting | [`methods/KALMAN.ipynb`](methods/KALMAN.ipynb) |
| ARIMA | Statistical per-phase forecasting | [`methods/ARIMA.ipynb`](methods/ARIMA.ipynb) |
| LSTM / BiLSTM | hidden=128, layers=6, Adam lr=1e-2, batch=64, 100 epochs | [`methods/LSTM.ipynb`](methods/LSTM.ipynb) |
| Mean-Retrieval | Top-k future-state averaging; no LLM | Ablation notebooks |
| Numeric-kNN | k=5, standardized numeric prefix features | [`knn_baseline.py`](iTransformer-DLinear-PatchTST/knn_baseline.py) |
| DLinear / PatchTST / iTransformer | seq/pred=5, epochs=30, seeds 0–2, lr=1e-3, batch=64 | [`run_all.py`](iTransformer-DLinear-PatchTST/run_all.py) |

The modern forecasting baselines follow their forecasting setup, while TrajRAG uses the variable blackout horizon defined by the controlled corruption protocol. See the protocol notes below before interpreting cross-method horizon differences.

---

## Published Table I

Mean Haversine error (km), reported as mean ± standard deviation.

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

Machine-readable aggregate values are stored in:

[`results/paper/table1_baselines.json`](results/paper/table1_baselines.json)

---

## How to reproduce

Experiments were conducted on a Dell Latitude 7450 laptop with an Intel Core Ultra 7 165U CPU, 32 GB RAM, integrated Intel Graphics, and Windows 11 Enterprise. GPU-backed runs are needed only for the relevant local-model / TSLib experiments.

### 0 — Smoke test

```bash
cd iTransformer-DLinear-PatchTST
python run_baseline.py --model iTransformer --epochs 2
python knn_baseline.py --runs 3
```

Smoke tests verify that the environment is working. They do **not** reproduce the full paper.

### 1 — Data preparation

Use:

[`fligh_radar_api.ipynb`](fligh_radar_api.ipynb)

to collect / reconstruct the allowed flight and weather inputs, then preprocess them into phase-specific JSON under `data/<route>/`.

For paper reproduction, use the chronological split and recorded corruption settings indexed in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

### 2 — TrajRAG → Table I

Run:

[`methods/RAG.ipynb`](methods/RAG.ipynb)

Main stages:

1. Build offline textual state summaries.
2. Embed and index phase-specific memory under `data/MULTI_ROUTE/EMBEDDINGS/`.
3. Reconstruct the 100 held-out test flights using the recorded paper corruption settings.
4. Write per-flight MHE results to:
   `data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json`.

### 3 — Classical baselines → Table I

| Notebook | Output |
| --- | --- |
| [`KALMAN.ipynb`](methods/KALMAN.ipynb) | `MEAN_HAVERSINE_KALMAN.json` |
| [`ARIMA.ipynb`](methods/ARIMA.ipynb) | Phase-wise MHE under `RESULTS/` |
| [`LSTM.ipynb`](methods/LSTM.ipynb) | `MEAN_HAVERSINE_LSTM.json` + BiLSTM block |

### 4 — TSLib + Numeric-kNN → Table I

```bash
cd iTransformer-DLinear-PatchTST
python adapt_data.py
python run_all.py --itransformer --bonus
python compile_results.py --compare-paper
```

CLI details:

[`iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md`](iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md)

### 5 — Extended controlled-benchmark analyses

| Paper item | Repository entry point |
| --- | --- |
| Fig. 2 — data efficiency | [`methods/LSTM.ipynb`](methods/LSTM.ipynb) + `run_convergence.py` |
| Table III — LLM backbone sensitivity | [`experiments/test_multiple_LLM/test_multiple_LLM.ipynb`](experiments/test_multiple_LLM/test_multiple_LLM.ipynb) |
| Table IV — mixed-route memory | [`data/BCN-MUC/RAG_for_new_trajectory.ipynb`](data/BCN-MUC/RAG_for_new_trajectory.ipynb) |
| Table V — ablations | `experiments/with_without_*` → [`experiments/generate_full_table.ipynb`](experiments/generate_full_table.ipynb) |
| K sensitivity | [`plots/plot_k_neighbors_vs_error.ipynb`](plots/plot_k_neighbors_vs_error.ipynb) |

---

## Real-world MLAT-supported case studies

Section VI of the paper complements the controlled benchmark with four operational GNSS-interference episodes for which FlightRadar24 reported **Terrestrial MLAT** positions.

The four reported cases are:

| Flight | Route | Interference window | Mean MLAT deviation (km) | Endpoint deviation (km) |
| --- | --- | ---: | ---: | ---: |
| A6-AQF | AMM → AUH | 103 min | 6.96 | 1.97 |
| TG941 | MXP → BKK | 46 min | 7.63 | 4.78 |
| CK602 | BUD → CKG | 71 min | 7.95 | 5.01 |
| QR8213 | DOH → MAD | 53 min | 6.95 | 6.36 |

Only fixes explicitly labelled **Terrestrial MLAT** are used as the independent consistency reference; ADS-B-labelled fixes inside the interference interval are discarded.

Because these cases depend on provider data and event-specific provenance, consult [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the released artifacts and exact mapping for:

- **Table VI** — case metadata,
- **Table VII** — reconstruction consistency metrics,
- **Fig. 3** — AMM→AUH reconstruction visualization.

Where source data cannot be redistributed, the repository documents the collection/query procedure rather than relicensing provider data.

---

## Verify against the paper

```bash
# Windows
type results\paper\table1_baselines.json

# Linux / macOS
cat results/paper/table1_baselines.json

cd iTransformer-DLinear-PatchTST
python compile_results.py --compare-paper
```

`--compare-paper` prints per-method differences against the released paper aggregate file.

For full TrajRAG comparison, first generate:

`data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json`

with the paper's recorded split and corruption settings.

---

## Protocol notes

1. **Paper blackout protocol** — The final paper uses randomized spoofing onset times and 5–15 missing steps. A fixed midpoint such as `SPOOF_INDEX = N // 2` is a diagnostic setting, not the paper protocol.

2. **TSLib horizon vs. TrajRAG horizon** — TSLib forecasting baselines use their fixed forecasting window, while TrajRAG follows the blackout horizon of the controlled benchmark. Interpret this distinction exactly as described in the paper.

3. **Chronological split** — The primary CDG→FCO evaluation uses 831 train/validation flights and 100 held-out flights. Additional baseline harnesses may have their own training partitioning; use the paper-matched configuration for reported comparisons.

4. **gpt-5.1** — Main TrajRAG results use `gpt-5.1` as specified in Sec. IV-G. Older development snapshots using other backbones will not reproduce the published aggregate values.

5. **Mean-Retrieval** — The retrieval-only baseline is reproduced through the matched *w/o LLM (mean retrieval)* pipeline used in the ablation study.

6. **Third-party services** — FlightRadar24 and Meteomatics data are governed by their respective terms. Public release of code or derived metadata does not grant redistribution rights for provider-restricted raw data.

---

## Reproducibility checklist

- [ ] Python 3.10 environment created
- [ ] `pip install -r requirements.txt`
- [ ] TSLib cloned where required
- [ ] `.env` configured only for the APIs being used
- [ ] CDG→FCO data prepared
- [ ] 831/100 chronological split verified
- [ ] Paper corruption settings loaded (randomized onset, 5–15-step blackout)
- [ ] Phase-specific embeddings built
- [ ] TrajRAG evaluation completed
- [ ] Classical and modern baselines completed
- [ ] `compile_results.py --compare-paper` checked
- [ ] Real-world case-study provenance checked separately where applicable

---

## More documentation

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — paper item → repository artifact map
- [`results/paper/README.md`](results/paper/README.md) — released aggregate snapshots
- [`iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md`](iTransformer-DLinear-PatchTST/RUN_EXPERIMENTS.md) — TSLib execution details

---

## Data and licensing notes

The repository is intended for research reproducibility.

Flight and weather data remain subject to the terms of the services from which they were obtained. No third-party dataset is relicensed by this repository. Do not commit `.env`, API credentials, provider-restricted raw data, or proprietary artifacts.

If you add a software `LICENSE` file, ensure that you are authorized to release all code covered by that license.
