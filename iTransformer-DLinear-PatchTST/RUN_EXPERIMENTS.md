# TSLib + KNN baselines (ICDM ADS 2026)

**Defaults match the paper (Table I baselines):** `seq_len=5`, `pred_len=5`, `epochs=30`, seeds `0–2`.
This folder reproduces **DLinear, PatchTST, iTransformer, and Numeric-kNN** only. TrajRAG and classical baselines are in `methods/`. See [README.md](../README.md).

All commands run from inside `iTransformer-DLinear-PatchTST/`.

**Quick run:** `python run_all.py --itransformer` (KNN + iTransformer; add `--bonus` for PatchTST/DLinear, `--all` for convergence plots).

---

## One-time setup

```bash
# 1. Clone TSLib (Time-Series-Library from THUML)
git clone --depth 1 https://github.com/thuml/Time-Series-Library.git TSLib

# 2. Install dependencies
pip install torch numpy pandas matplotlib scikit-learn reformer_pytorch
```

---

## Step 1 — Convert data

```bash
python adapt_data.py
```

Reads all 6 route folders from `../data/`, outputs `adapted_flights.csv`.
Prints a summary: total virtual flights, rows, steps per flight, phase split.

---

## Step 2 — Smoke test (synthetic data, no real data needed)

```bash
python run_baseline.py --model iTransformer --epochs 2
```

Should complete in <60 sec and print a non-NaN MHE.  If this fails, debug TSLib import.

---

## Step 3 — iTransformer on real data (MANDATORY, 3 seeds)

```bash
python run_baseline.py --model iTransformer --csv adapted_flights.csv \
    --seq_len 5 --pred_len 5 --epochs 30 --seed 0

python run_baseline.py --model iTransformer --csv adapted_flights.csv \
    --seq_len 5 --pred_len 5 --epochs 30 --seed 1

python run_baseline.py --model iTransformer --csv adapted_flights.csv \
    --seq_len 5 --pred_len 5 --epochs 30 --seed 2
```

Results saved to `results/iTransformer_seed0.json`, `_seed1.json`, `_seed2.json`.

---

## Step 4 — PatchTST and DLinear (bonus, same 3 seeds)

```bash
for seed in 0 1 2; do
  python run_baseline.py --model PatchTST --csv adapted_flights.csv \
      --seq_len 5 --pred_len 5 --epochs 30 --seed $seed
  python run_baseline.py --model DLinear  --csv adapted_flights.csv \
      --seq_len 5 --pred_len 5 --epochs 30 --seed $seed
done
```

---

## Step 5 — KNN baseline (MANDATORY)

```bash
python knn_baseline.py --runs 3
```

Uses the same JSON data directly (no adapted_flights.csv needed).
Results saved to `results/KNN_results.json`.
Expected: std ≈ 0 (KNN is deterministic), confirms reproducibility.

---

## Step 6 — Convergence plot for iTransformer

```bash
python run_convergence.py --csv adapted_flights.csv --epochs 30 --seeds 3
```

Outputs:
- `results/convergence_iTransformer.csv`
- `results/convergence_iTransformer.png`

---

## Step 7 — Compile all results into table

```bash
python compile_results.py
```

Reads `results/*.json` + original TrajRAG numbers from `../data/MULTI_ROUTE/RESULTS/`.
Prints a Markdown table and saves to `results/comparison_table.md`.

---

## Protocol notes for the paper

| Aspect | Value |
|---|---|
| Train/test split | First 83% / last 17% of flights, **chronological** |
| Blackout definition | First N//2 steps = observed prefix; last N-N//2 = predicted |
| Metric | Mean Haversine Error (km), reported per phase (takeoff/cruise/landing) + overall |
| TSLib seq\_len | 5 (covers cruise/landing; takeoff phases <10 steps are skipped) |
| TSLib pred\_len | 5 |
| Seeds | 0, 1, 2 → report mean ± std |
| KNN k | 5 neighbours |

**Asymmetry to disclose:** TSLib models predict with a fixed `pred_len=5` window,
while TrajRAG uses a variable window (N-N//2).  KNN uses the same variable window
as TrajRAG and is therefore directly comparable.
