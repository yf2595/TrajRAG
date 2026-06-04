"""
run_all.py  —  end-to-end experiment runner + paper-style plots.

TRAINING REQUIREMENTS:
  KNN          — NO training. Indexes lat/lon positions, queries instantly.
  iTransformer — YES, supervised (~100 epochs with early stopping, ~10-30 min).
  PatchTST     — YES, same as iTransformer.
  DLinear      — YES, but trivially fast (linear model).

USAGE:
  python run_all.py --itransformer --bonus   # all 3 models + KNN
  python run_all.py --itransformer           # iTransformer + KNN only
  python run_all.py --all                    # + convergence plot
  python run_all.py --plots_only   # skip experiments, regenerate plots
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
DATA_ROOT = os.path.join(HERE, "..", "data")
SAMPLE_EFF_DIR = os.path.join(HERE, "..", "experiments", "sample_efficiency")

PHASES_JSON = ["take_off", "cruising", "landing"]
PHASES_LABEL = ["takeoff", "cruise", "landing"]
PHASE_DISPLAY = ["Take-off", "Cruising", "Landing"]

MSF_KEY = "Mean Similar Flight"
KNN_KEY = "KNN (k=5)"

COLORS = {
    "KALMAN":        "#7f7f7f",
    "ARIMA":         "#bcbd22",
    "LSTM":          "#ff7f0e",
    "BiLSTM":        "#e377c2",
    MSF_KEY:         "#17becf",
    KNN_KEY:         "#9467bd",
    "iTransformer":  "#2ca02c",
    "PatchTST":      "#d62728",
    "DLinear":       "#8c564b",
    "TrajRAG":       "#1f77b4",
}

METHOD_ORDER = [
    "KALMAN", "ARIMA", "BiLSTM", "LSTM",
    MSF_KEY, KNN_KEY,
    "DLinear", "PatchTST", "iTransformer", "TrajRAG",
]


# ---
# Helpers
# ---
def run(cmd: list, **kw):
    result = subprocess.run([sys.executable] + cmd, cwd=HERE, **kw)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


# ---
# Experiment steps
# ---
def step_adapt_data():
    csv = os.path.join(HERE, "adapted_flights.csv")
    if os.path.exists(csv):
        print("[skip] adapted_flights.csv already exists.")
        return csv
    print("\n── Step 1: Converting data ──────────────────────────────")
    run(["adapt_data.py"])
    return csv


def step_knn(force=False):
    out = os.path.join(RESULTS_DIR, "KNN_results.json")
    if os.path.exists(out) and not force:
        print("[skip] KNN_results.json already exists.")
        return
    print("\n── Step 2: KNN baseline (no training) ──────────────────")
    run(["knn_baseline.py", "--runs", "3"])


def step_train_model(model, csv, seeds=(0, 1, 2),
                     seq_len=5, pred_len=5, epochs=30, force=False):
    print(f"\n── Training {model} "
          f"(seq={seq_len}, pred={pred_len}, "
          f"epochs≤{epochs}) ──────────────")
    for seed in seeds:
        out = os.path.join(RESULTS_DIR, f"{model}_seed{seed}.json")
        if os.path.exists(out) and not force:
            print(f"   [skip] {model}_seed{seed}.json already exists.")
            continue
        run([
            "run_baseline.py",
            "--model", model,
            "--csv", csv,
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--epochs", str(epochs),
            "--seed", str(seed),
        ])


def step_convergence(csv, epochs=30, force=False):
    out = os.path.join(RESULTS_DIR, "convergence_iTransformer.csv")
    if os.path.exists(out) and not force:
        print("[skip] convergence_iTransformer.csv already exists.")
        return
    print("\n── Step: iTransformer convergence ───────────────────────")
    run(["run_convergence.py", "--csv", csv,
         "--epochs", str(epochs), "--seeds", "3"])


def step_compile():
    print("\n── Compiling results table ───────────────────────────────")
    run(["compile_results.py"])


# ---
# Result loaders
# ---
def load_original_stats():
    path = os.path.join(
        DATA_ROOT, "CDG-FCO", "RESULTS", "BOXPLOT_STATS_ALL_METHODS.csv"
    )
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    name_map = {
        "KALMAN": "KALMAN", "ARIMA": "ARIMA",
        "LSTM": "LSTM", "BILSTM": "BiLSTM",
        "MEAN_SIMILAR_FLIGHT": MSF_KEY, "RAG": "TrajRAG",
    }
    phase_map = {
        "take_off": "takeoff", "cruising": "cruise",
        "landing": "landing",
    }
    result = {}
    for _, row in df.iterrows():
        method = name_map.get(row["Method"], row["Method"])
        phase = phase_map.get(row["Phase"], row["Phase"])
        if method not in result:
            result[method] = {}
        result[method][phase] = {
            "mean": float(row["Mean"]),
            "std": float(row["Std"]),
        }
    for method, phases in result.items():
        vals = [v["mean"] for v in phases.values()
                if not is_nan(v["mean"])]
        result[method]["overall"] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float("nan"),
        }
    return result


def load_knn_results():
    path = os.path.join(RESULTS_DIR, "KNN_results.json")
    if not os.path.exists(path):
        return None
    data = load_json(path)
    ph = data.get("phases", {})
    return {
        label: {
            "mean": ph.get(label, {}).get("mean", float("nan")),
            "std":  ph.get(label, {}).get("std",  float("nan")),
        }
        for label in PHASES_LABEL + ["overall"]
    }


def load_tslib_results(model):
    by_phase = {p: [] for p in PHASES_LABEL + ["overall"]}
    for seed in range(3):
        path = os.path.join(RESULTS_DIR, f"{model}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        data = load_json(path)
        for ph in PHASES_LABEL + ["overall"]:
            v = data.get(ph, float("nan"))
            if not is_nan(v):
                by_phase[ph].append(v)
    if not any(by_phase.values()):
        return None
    return {
        ph: {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
        }
        for ph, vals in by_phase.items()
    }


def load_sample_efficiency():
    rag_path = os.path.join(SAMPLE_EFF_DIR, "results_RAG.json")
    lstm_path = os.path.join(SAMPLE_EFF_DIR, "results_LSTM.json")
    if not (os.path.exists(rag_path) and os.path.exists(lstm_path)):
        return None, None

    def parse_rag(obj):
        if "results" not in obj:
            return {
                int(k): {p: float(v[p]["error"]) for p in PHASES_JSON}
                for k, v in obj.items()
            }
        return {
            int(r["sample_size"]): {
                p: float(r["RAG"][p]) for p in PHASES_JSON
            }
            for r in obj["results"]
        }

    def parse_lstm(obj):
        return {
            int(r["sample_size"]): {
                p: float(r["LSTM"][p]) for p in PHASES_JSON
            }
            for r in obj.get("results", [])
        }

    return parse_rag(load_json(rag_path)), parse_lstm(load_json(lstm_path))


def load_itransformer_convergence():
    path = os.path.join(RESULTS_DIR, "convergence_iTransformer.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    agg = (
        df.groupby("n_train")["mhe"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("n_train")
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


# ---
# Plots
# ---
def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot.")
        return None


def plot_comparison_bars(all_methods):
    plt = _get_mpl()
    if plt is None:
        return

    present = [m for m in METHOD_ORDER if m in all_methods]
    n = len(present)
    x = np.arange(3)
    width = 0.8 / n
    offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n)

    _, ax = plt.subplots(figsize=(max(10, n * 1.3), 6))
    for i, method in enumerate(present):
        means, stds = [], []
        for ph in PHASES_LABEL[:3]:
            entry = all_methods[method].get(ph, {})
            m = entry.get("mean", float("nan"))
            s = entry.get("std", float("nan"))
            means.append(0 if is_nan(m) else m)
            stds.append(0 if is_nan(s) else s)
        color = COLORS.get(method, f"C{i}")
        edge = "black" if method == "TrajRAG" else color
        lw = 1.5 if method == "TrajRAG" else 0.5
        ax.bar(
            x + offsets[i], means, width,
            yerr=stds, capsize=3,
            label=method, color=color,
            edgecolor=edge, linewidth=lw,
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )

    ax.set_xlabel("Flight phase", fontsize=13)
    ax.set_ylabel("Mean Haversine Error (km)", fontsize=13)
    ax.set_title(
        "Trajectory Reconstruction — Method Comparison", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_DISPLAY, fontsize=12)
    ax.legend(
        fontsize=9, ncol=2, framealpha=0.9,
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_comparison_panels(all_methods):
    plt = _get_mpl()
    if plt is None:
        return

    present = [m for m in METHOD_ORDER if m in all_methods]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Mean Haversine Error per Phase (km)", fontsize=14, y=1.02
    )

    for ax, ph, ph_disp in zip(axes, PHASES_LABEL[:3], PHASE_DISPLAY):
        labels, means, stds, colors = [], [], [], []
        for method in present:
            entry = all_methods[method].get(ph, {})
            m = entry.get("mean", float("nan"))
            s = entry.get("std", float("nan"))
            if is_nan(m):
                continue
            labels.append(method)
            means.append(m)
            stds.append(0 if is_nan(s) else s)
            colors.append(COLORS.get(method, "gray"))

        y = np.arange(len(labels))
        bars = ax.barh(
            y, means, xerr=stds, capsize=3,
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw={"elinewidth": 1},
        )
        for bar, label in zip(bars, labels):
            if label == "TrajRAG":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("MHE (km)", fontsize=11)
        ax.set_title(ph_disp, fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_panels.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_convergence(rag_data, lstm_data, itrans_agg):
    plt = _get_mpl()
    if plt is None:
        return
    if rag_data is None and lstm_data is None and itrans_agg is None:
        print("  [skip] No convergence data available.")
        return

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 8), sharex=False,
        gridspec_kw={"hspace": 0.35},
    )
    fig.suptitle(
        "Sample Efficiency — Training Data vs MHE", fontsize=14
    )

    for idx, (ph_json, ph_disp) in enumerate(
        zip(PHASES_JSON, PHASE_DISPLAY)
    ):
        ax = axes[idx]
        if rag_data:
            sizes = sorted(rag_data.keys())
            ax.plot(
                sizes,
                [rag_data[s][ph_json] for s in sizes],
                marker="o", linewidth=2,
                color=COLORS["TrajRAG"], label="TrajRAG (ours)",
            )
        if lstm_data and rag_data:
            common = sorted(set(rag_data) & set(lstm_data))
            ax.plot(
                common,
                [lstm_data[s][ph_json] for s in common],
                marker="s", linewidth=2, linestyle="--",
                color=COLORS["LSTM"], label="LSTM",
            )
        if itrans_agg is not None:
            ax.plot(
                itrans_agg["n_train"], itrans_agg["mean"],
                marker="^", linewidth=2, linestyle="-.",
                color=COLORS["iTransformer"], label="iTransformer",
            )
            ax.fill_between(
                itrans_agg["n_train"],
                itrans_agg["mean"] - itrans_agg["std"],
                itrans_agg["mean"] + itrans_agg["std"],
                alpha=0.15, color=COLORS["iTransformer"],
            )
        ax.set_title(ph_disp, fontsize=11)
        ax.set_ylabel("MHE (km)", fontsize=10)
        ax.grid(True, linewidth=0.4, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel(
        "Training set size (# flights)", fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "convergence_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ---
# Summary table
# ---
def print_summary(all_methods):
    present = [m for m in METHOD_ORDER if m in all_methods]
    sep = "=" * 76
    hdr = (
        f"  {'Model':<22s}"
        f" {'Takeoff':>12s}"
        f" {'Cruise':>12s}"
        f" {'Landing':>12s}"
        f" {'Overall':>12s}"
    )
    print("\n" + sep)
    print("  RESULTS — Mean Haversine Error (km),  mean ± std over 3 seeds")
    print(sep)
    print(hdr)
    print("-" * 76)

    def cell(entry):
        m = entry.get("mean", float("nan"))
        s = entry.get("std", float("nan"))
        if is_nan(m):
            return "      —     "
        if is_nan(s) or s < 0.01:
            return f"  {m:6.2f}      "
        return f"  {m:5.2f}±{s:4.2f} "

    for method in present:
        d = all_methods[method]
        row = (
            f"  {method:<22s}"
            f"{cell(d.get('takeoff', {})):>12s}"
            f"{cell(d.get('cruise', {})):>12s}"
            f"{cell(d.get('landing', {})):>12s}"
            f"{cell(d.get('overall', {})):>12s}"
        )
        print(row)
    print(sep)


# ---
# Main
# ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itransformer", action="store_true",
                    help="Train iTransformer (needs TSLib)")
    ap.add_argument("--bonus", action="store_true",
                    help="Also train PatchTST and DLinear")
    ap.add_argument("--convergence", action="store_true",
                    help="Run iTransformer convergence experiment")
    ap.add_argument("--all", dest="run_all", action="store_true",
                    help="Run everything")
    ap.add_argument("--plots_only", action="store_true",
                    help="Skip experiments, regenerate plots only")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if result files exist")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=5)
    ap.add_argument("--pred_len", type=int, default=5)
    args = ap.parse_args()

    if args.run_all:
        args.itransformer = True
        args.bonus = True
        args.convergence = True

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  GPS-Spoofing Correction — Experiment Runner")
    print("=" * 60)
    itf_msg = (
        f"YES — up to {args.epochs} epochs (early stop)"
        if args.itransformer else "skip (use --itransformer)"
    )
    bon_msg = "YES" if args.bonus else "skip (use --bonus)"
    conv_msg = "YES" if args.convergence else "skip (use --convergence)"
    print("  KNN:           always (no training)")
    print(f"  iTransformer:  {itf_msg}")
    print(f"  PatchTST/DLin: {bon_msg}")
    print(f"  Convergence:   {conv_msg}")
    print("=" * 60)

    csv = None
    if not args.plots_only:
        if args.itransformer or args.bonus or args.convergence:
            csv = step_adapt_data()

        step_knn(force=args.force)

        if args.itransformer:
            step_train_model(
                "iTransformer", csv,
                seq_len=args.seq_len, pred_len=args.pred_len,
                epochs=args.epochs, force=args.force,
            )
        if args.bonus:
            step_train_model(
                "PatchTST", csv,
                seq_len=args.seq_len, pred_len=args.pred_len,
                epochs=args.epochs, force=args.force,
            )
            step_train_model(
                "DLinear", csv,
                seq_len=args.seq_len, pred_len=args.pred_len,
                epochs=args.epochs, force=args.force,
            )
        if args.convergence:
            if csv is None:
                csv = step_adapt_data()
            step_convergence(
                csv, epochs=args.epochs, force=args.force
            )
        step_compile()

    # ── Assemble results ──────────────────────────────────────────
    print("\n── Generating plots ─────────────────────────────────────")
    all_methods = {}
    all_methods.update(load_original_stats())

    knn = load_knn_results()
    if knn:
        all_methods[KNN_KEY] = knn

    for model_name in ["iTransformer", "PatchTST", "DLinear"]:
        res = load_tslib_results(model_name)
        if res:
            all_methods[model_name] = res

    if not all_methods:
        print("[WARN] No results to plot. Run experiments first.")
        return

    plot_comparison_bars(all_methods)
    plot_comparison_panels(all_methods)

    rag_data, lstm_data = load_sample_efficiency()
    itrans_agg = load_itransformer_convergence()
    plot_convergence(rag_data, lstm_data, itrans_agg)

    print_summary(all_methods)
    print(f"\nAll plots saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
