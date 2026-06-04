"""
Compile all baseline results into a single Markdown comparison table.

Reads from results/*.json and the original TrajRAG numbers from
data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json (or CDG-FCO/RESULTS/BOXPLOT_STATS_ALL_METHODS.csv).

Usage:
    python compile_results.py
    python compile_results.py --results_dir results --out results/comparison_table.md
"""
import argparse
import json
import math
import os
import statistics

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PAPER_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "paper"
)
PHASES = ["takeoff", "cruise", "landing"]
PHASE_DISPLAY = {"takeoff": "Takeoff (km)", "cruise": "Cruise (km)",
                 "landing": "Landing (km)", "overall": "Overall (km)"}


def fmt(mean, std=None):
    if math.isnan(mean):
        return "   —   "
    if std is None or math.isnan(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


# --------------------------------------------------------------------------- #
# Load TrajRAG original results
# --------------------------------------------------------------------------- #
def load_trajrag_results():
    """
    Load TrajRAG numbers from data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json
    which stores per-flight MHE as [[error, fr24_id], ...] per phase.
    Falls back to BOXPLOT_STATS_ALL_METHODS.csv if the JSON is missing.
    """
    # Primary: MULTI_ROUTE JSON
    json_path = os.path.join(DATA_ROOT, "MULTI_ROUTE", "RESULTS",
                             "MEAN_HAVERSINE_RAG.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        rag = data.get("RAG", {})
        result = {}
        all_vals = []
        for ph_json, ph_label in [("take_off", "takeoff"),
                                   ("cruising", "cruise"),
                                   ("landing", "landing")]:
            vals = [item[0] for item in rag.get(ph_json, []) if isinstance(item, list)]
            result[ph_label] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                                 "std": float(np.std(vals)) if vals else float("nan"),
                                 "n": len(vals)}
            all_vals.extend(vals)
        result["overall"] = {"mean": float(np.mean(all_vals)) if all_vals else float("nan"),
                              "std": float(np.std(all_vals)) if all_vals else float("nan"),
                              "n": len(all_vals)}
        return result

    # Fallback: CDG-FCO BOXPLOT_STATS CSV
    csv_path = os.path.join(DATA_ROOT, "CDG-FCO", "RESULTS",
                             "BOXPLOT_STATS_ALL_METHODS.csv")
    if os.path.exists(csv_path):
        import csv as csvmod
        result = {}
        all_means = []
        with open(csv_path) as f:
            for row in csvmod.DictReader(f):
                if row.get("Method", "").upper() != "RAG":
                    continue
                ph_map = {"take_off": "takeoff", "cruising": "cruise",
                          "landing": "landing"}
                ph = ph_map.get(row.get("Phase", "").lower())
                if ph:
                    m = float(row.get("Mean", "nan"))
                    s = float(row.get("Std", "nan"))
                    result[ph] = {"mean": m, "std": s, "n": None}
                    if not math.isnan(m):
                        all_means.append(m)
        if all_means:
            result["overall"] = {"mean": float(np.mean(all_means)),
                                  "std": float("nan"), "n": None}
        return result if result else None

    return None


# --------------------------------------------------------------------------- #
# Load baseline JSON results from results/
# --------------------------------------------------------------------------- #
def load_model_results(results_dir: str):
    """
    Scan results_dir for JSON files. Supported formats:
      - <Model>_seed<N>.json  → {"takeoff": x, "cruise": x, "landing": x, "overall": x}
      - KNN_results.json      → {"phases": {"takeoff": {"mean":..,"std":..}, ...}}
    Returns dict: model_name → {phase: {mean, std}}
    """
    models = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        # KNN results
        if "KNN" in fname and "phases" in data:
            ph_data = {}
            all_means = []
            for ph in PHASES:
                entry = data["phases"].get(ph, {})
                m = entry.get("mean", float("nan"))
                s = entry.get("std", float("nan"))
                ph_data[ph] = {"mean": m, "std": s}
                if not math.isnan(m):
                    all_means.append(m)
            overall = data["phases"].get("overall", {})
            ph_data["overall"] = {"mean": overall.get("mean", float("nan")),
                                   "std": overall.get("std", float("nan"))}
            k = data.get("k", 5)
            models[f"KNN (k={k})"] = ph_data
            continue

        # Per-seed results: {takeoff: x, cruise: x, landing: x, overall: x}
        if all(ph in data for ph in PHASES):
            # figure out model name from filename
            name = fname.replace("_results.json", "").replace(".json", "")
            # strip _seed suffix → group runs
            base = name.rsplit("_seed", 1)[0]
            if base not in models:
                models[base] = {ph: [] for ph in PHASES + ["overall"]}
            for ph in PHASES + ["overall"]:
                v = data.get(ph, float("nan"))
                if not math.isnan(v):
                    models[base][ph].append(v)

    # Aggregate per-seed lists to mean/std
    aggregated = {}
    for name, data in models.items():
        if isinstance(list(data.values())[0], list):
            agg = {}
            for ph in PHASES + ["overall"]:
                vals = [v for v in data.get(ph, []) if not math.isnan(v)]
                agg[ph] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": float(np.std(vals)) if vals else float("nan"),
                }
            aggregated[name] = agg
        else:
            aggregated[name] = data
    return aggregated


# --------------------------------------------------------------------------- #
# Render table
# --------------------------------------------------------------------------- #
def load_paper_table1():
    """Load published Table I aggregates from results/paper/table1_baselines.json."""
    path = os.path.join(PAPER_RESULTS_DIR, "table1_baselines.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for name, phases in data.get("methods", {}).items():
        out[name] = {
            ph: {"mean": phases[ph]["mean"], "std": phases[ph]["std"]}
            for ph in PHASES
            if ph in phases
        }
        vals = [
            phases[ph]["mean"]
            for ph in PHASES
            if ph in phases and not math.isnan(phases[ph]["mean"])
        ]
        if vals:
            out[name]["overall"] = {
                "mean": float(np.mean(vals)),
                "std": float("nan"),
            }
    return out


def compare_to_paper(rerun: dict, paper: dict, rtol: float = 0.15) -> str:
    """Markdown summary: Δmean (km) vs published Table I for overlapping methods."""
    lines = [
        "\n## Comparison vs. published Table I (`results/paper/table1_baselines.json`)\n",
        "| Model | Phase | Rerun mean | Paper mean | Delta (km) |",
        "|-------|-------|---------|---------|--------|",
    ]
    name_map = {
        "KNN (k=5)": "Numeric-kNN",
        "KNN": "Numeric-kNN",
    }
    for model, ph_data in sorted(rerun.items()):
        paper_name = name_map.get(model, model)
        if paper_name not in paper:
            continue
        for ph in PHASES:
            r = ph_data.get(ph, {})
            p = paper[paper_name].get(ph, {})
            rm = r.get("mean", float("nan")) if isinstance(r, dict) else float("nan")
            pm = p.get("mean", float("nan")) if isinstance(p, dict) else float("nan")
            if math.isnan(rm) or math.isnan(pm):
                continue
            delta = rm - pm
            lines.append(
                f"| {model} | {ph} | {rm:.2f} | {pm:.2f} | {delta:+.2f} |"
            )
    lines.append(
        "\n*Large delta expected if sample data, different LLM snapshot, or TSLib "
        "fixed `pred_len` protocol. TrajRAG/KNN should match when full data "
        "and `MEAN_HAVERSINE_RAG.json` are present.*\n"
    )
    return "\n".join(lines)


def render_table(all_models: dict) -> str:
    cols = ["Takeoff (km)", "Cruise (km)", "Landing (km)", "Overall (km)"]
    ph_keys = ["takeoff", "cruise", "landing", "overall"]

    # header
    lines = []
    header = "| {:<22s} | {:>16s} | {:>16s} | {:>16s} | {:>16s} |".format(
        "Model", *cols)
    sep = "|" + "|".join(["-" * 24, "-" * 18, "-" * 18, "-" * 18, "-" * 18]) + "|"
    lines.append(header)
    lines.append(sep)

    for model_name, ph_data in all_models.items():
        cells = []
        for ph in ph_keys:
            entry = ph_data.get(ph, {})
            if isinstance(entry, dict):
                m = entry.get("mean", float("nan"))
                s = entry.get("std", float("nan"))
                cells.append(fmt(m, s))
            else:
                cells.append(fmt(float(entry)))
        row = "| {:<22s} | {:>16s} | {:>16s} | {:>16s} | {:>16s} |".format(
            model_name, *cells)
        lines.append(row)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "comparison_table.md"))
    ap.add_argument(
        "--compare-paper",
        action="store_true",
        help="Append Δ vs. results/paper/table1_baselines.json",
    )
    args = ap.parse_args()

    # Collect all models
    all_models = {}

    # TrajRAG (original results)
    rag = load_trajrag_results()
    if rag:
        all_models["TrajRAG"] = rag
    else:
        print("[WARN] Could not load TrajRAG original results.")

    # Baselines from results/
    if os.path.isdir(args.results_dir):
        baselines = load_model_results(args.results_dir)
        all_models.update(baselines)
    else:
        print(f"[WARN] Results directory not found: {args.results_dir}")

    if not all_models:
        print("[ERROR] No results to compile. Run experiments first.")
        return

    table = render_table(all_models)

    print("\n" + "=" * 100)
    print("  COMPARISON TABLE — Mean Haversine Error")
    print("=" * 100)
    print(table)
    print("=" * 100)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    paper_section = ""
    if args.compare_paper:
        paper = load_paper_table1()
        if paper:
            paper_section = compare_to_paper(all_models, paper)
            print(paper_section)
        else:
            print("[WARN] Paper snapshot not found at results/paper/table1_baselines.json")

    with open(args.out, "w") as f:
        f.write("# Trajectory Reconstruction — Baseline Comparison\n\n")
        f.write("Mean Haversine Error (km), mean +/- std over seeds.\n\n")
        f.write(table + "\n")
        if paper_section:
            f.write(paper_section)
    print(f"\nTable saved -> {args.out}")

    # Also save a machine-readable JSON summary
    json_out = args.out.replace(".md", ".json")
    with open(json_out, "w") as f:
        json.dump(all_models, f, indent=2)
    print(f"JSON summary -> {json_out}")


if __name__ == "__main__":
    main()
