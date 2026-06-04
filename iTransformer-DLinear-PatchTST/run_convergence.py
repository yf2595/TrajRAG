"""
Learning-curve (convergence) experiment for iTransformer.

Mirrors the sample-efficiency experiment done for LSTM in experiments/sample_efficiency/.
For each training-data fraction, trains iTransformer for 30 epochs and evaluates
on the FULL fixed test set.  Repeats for 3 seeds → mean ± std per fraction point.

Outputs:
  results/convergence_iTransformer.csv   — raw numbers
  results/convergence_iTransformer.png   — plot with error bars

Usage:
    python run_convergence.py --csv adapted_flights.csv
    python run_convergence.py --csv adapted_flights.csv --epochs 30 --seeds 3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Use the same harness components
from configs import Configs
from data import (FEATURES, TARGET_IDX, PHASES, Scaler,
                  TrajReconstructionDataset, haversine_km, make_synthetic_csv)
from models_wrapper import build_model
from run_baseline import chronological_split, train_one, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def build_configs(seq_len, pred_len, n_feats):
    return Configs(
        seq_len=seq_len, pred_len=pred_len,
        enc_in=n_feats, dec_in=n_feats, c_out=n_feats,
    )


def run_fraction(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 frac: float, seed: int,
                 seq_len: int, pred_len: int, epochs: int,
                 batch_size: int, lr: float, device: str) -> float:
    """Train on `frac` of train_df (first frac% of flight_ids) → overall MHE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Take the first `frac` fraction of training flight_ids (chronological)
    train_ids = sorted(train_df["flight_id"].unique())
    cut = max(1, int(len(train_ids) * frac))
    subset_ids = set(train_ids[:cut])
    subset_df = train_df[train_df["flight_id"].isin(subset_ids)]

    scaler = Scaler().fit(subset_df[FEATURES].to_numpy(dtype=float))
    tr_ds = TrajReconstructionDataset(subset_df, seq_len, pred_len, scaler)
    te_ds = TrajReconstructionDataset(test_df, seq_len, pred_len, scaler)

    if len(tr_ds) == 0 or len(te_ds) == 0:
        print(f"  [skip frac={frac:.1f} seed={seed}] "
              f"train_wins={len(tr_ds)} test_wins={len(te_ds)}")
        return float("nan")

    model = build_model("iTransformer",
                        build_configs(seq_len, pred_len, len(FEATURES))).to(device)
    train_one(model, DataLoader(tr_ds, batch_size, shuffle=True),
              device, epochs, lr)
    res = evaluate(model, DataLoader(te_ds, batch_size), device, scaler)
    return res["overall"]


def main():
    ap = argparse.ArgumentParser(description="iTransformer convergence experiment")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seq_len", type=int, default=5)
    ap.add_argument("--pred_len", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_frac", type=float, default=0.83)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seq={args.seq_len}  pred={args.pred_len}  "
          f"epochs={args.epochs}  seeds={args.seeds}")

    csv = args.csv or make_synthetic_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "synthetic_flights.csv"))
    df = pd.read_csv(csv)
    train_df, test_df = chronological_split(df, args.train_frac)

    n_train_flights = train_df["flight_id"].nunique()
    n_test_flights = test_df["flight_id"].nunique()
    print(f"Train flights={n_train_flights}  Test flights={n_test_flights}")

    results = []   # rows: {frac, n_train, seed, mhe}

    for frac in FRACTIONS:
        n_used = max(1, int(n_train_flights * frac))
        print(f"\n--- fraction={frac:.1f}  n_train_flights={n_used} ---")
        for seed in range(args.seeds):
            mhe = run_fraction(train_df, test_df, frac, seed,
                               args.seq_len, args.pred_len, args.epochs,
                               args.batch_size, args.lr, device)
            print(f"  seed={seed}  MHE={mhe:.3f} km")
            results.append({"fraction": frac, "n_train": n_used,
                             "seed": seed, "mhe": mhe})

    df_res = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_out = os.path.join(RESULTS_DIR, "convergence_iTransformer.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nRaw results saved → {csv_out}")

    # Aggregate mean ± std per fraction
    agg = (df_res.groupby("n_train")["mhe"]
           .agg(mean="mean", std="std")
           .reset_index())
    agg["std"] = agg["std"].fillna(0.0)

    print(f"\n{'n_train':>10s}  {'MHE mean':>10s}  {'MHE std':>10s}")
    print("-" * 36)
    for _, row in agg.iterrows():
        print(f"{int(row.n_train):>10d}  {row['mean']:>10.3f}  {row['std']:>10.3f}")

    # Save aggregated to JSON too
    agg_path = os.path.join(RESULTS_DIR, "convergence_iTransformer_agg.json")
    agg.to_json(agg_path, orient="records", indent=2)

    # Plot
    _make_plot(agg, n_train_flights, args)


def _make_plot(agg: pd.DataFrame, n_train_total: int, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(agg["n_train"], agg["mean"], yerr=agg["std"],
                fmt="o-", color="#1f77b4", linewidth=2, markersize=7,
                capsize=5, capthick=1.5, label="iTransformer")

    ax.set_xlabel("Number of training flights", fontsize=13)
    ax.set_ylabel("Mean Haversine Error (km)", fontsize=13)
    ax.set_title("iTransformer — Training Data Convergence", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Annotate with exact values
    for _, row in agg.iterrows():
        ax.annotate(f"{row['mean']:.2f}",
                    xy=(row["n_train"], row["mean"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color="#444444")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "convergence_iTransformer.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    main()
