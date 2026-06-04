"""
Pure physical KNN baseline for GNSS-blackout trajectory reconstruction.

This replaces TrajRAG's text-embedding retrieval with spatial position matching:
no text summaries, no OpenAI embeddings — just raw lat/lon trajectories.

Protocol (identical to TrajRAG's evaluation):
  * For each test flight-phase, observe the first N//2 steps (prefix).
  * Find the 5 training flight-phases (same phase label) whose prefixes are
    most similar by mean point-wise Haversine distance.
  * Predict each blackout step as the coordinate-weighted mean of the 5 neighbours'
    continuations at that step.
  * Evaluate with mean Haversine error (MHE, km) over the predicted horizon.

Train/test split: chronological by real flight (same 83%/17% as TrajRAG).

Usage:
    python knn_baseline.py
    python knn_baseline.py --data_root ../data --k 5 --runs 3
"""
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

ROUTES = ["MRS-MUC", "NTE-MUC", "BCN-MUC", "BCN-CDG", "CDG-FCO", "BOD-VCE"]
PHASES = ["take_off", "cruising", "landing"]
PHASE_LABEL = {"take_off": "takeoff", "cruising": "cruise", "landing": "landing"}
MIN_STEPS = 4
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --------------------------------------------------------------------------- #
# Haversine
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts[:19], fmt[:19])
        except Exception:
            pass
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def load_all_phases(data_root: str) -> List[dict]:
    """Load every (flight, phase) entry across all routes."""
    entries = []
    for route in ROUTES:
        path = os.path.join(data_root, route,
                            "flight_data_with_minutes_since_start.json")
        if not os.path.exists(path):
            print(f"[WARN] Not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            flights = json.load(f)
        for flight in flights:
            meta = flight.get("flight_metadata", {})
            fr24_id = meta.get("fr24_id", "unknown")
            for phase in PHASES:
                pts = flight.get(phase, [])
                if len(pts) < MIN_STEPS:
                    continue
                pts_sorted = sorted(pts, key=lambda p: p.get("timestamp", ""))
                first_ts = parse_ts(pts_sorted[0].get("timestamp", ""))
                latlon = [(float(p["lat"]), float(p["lon"])) for p in pts_sorted]
                entries.append({
                    "fr24_id": fr24_id,
                    "route": route,
                    "phase": phase,
                    "phase_label": PHASE_LABEL[phase],
                    "first_ts": first_ts,
                    "latlon": latlon,        # list of (lat, lon)
                    "n": len(latlon),
                })
    return entries


def chronological_split(entries: List[dict], train_frac: float = 0.83):
    """Split entries at the real-flight level, chronologically."""
    # Sort by (first_ts, fr24_id) for stability
    entries_sorted = sorted(entries, key=lambda e: (e["first_ts"], e["fr24_id"]))

    # Unique real flights in order
    seen = {}
    ordered_flights = []
    for e in entries_sorted:
        fid = e["fr24_id"]
        if fid not in seen:
            seen[fid] = e["first_ts"]
            ordered_flights.append(fid)

    cut = int(len(ordered_flights) * train_frac)
    train_ids = set(ordered_flights[:cut])
    test_ids = set(ordered_flights[cut:])

    train = [e for e in entries_sorted if e["fr24_id"] in train_ids]
    test = [e for e in entries_sorted if e["fr24_id"] in test_ids]
    return train, test


# --------------------------------------------------------------------------- #
# KNN retrieval
# --------------------------------------------------------------------------- #
def prefix_dist(query_prefix: List[Tuple[float, float]],
                train_prefix: List[Tuple[float, float]]) -> float:
    """Mean point-wise Haversine over the overlapping prefix steps."""
    overlap = min(len(query_prefix), len(train_prefix))
    if overlap == 0:
        return float("inf")
    total = sum(haversine_km(query_prefix[i][0], query_prefix[i][1],
                             train_prefix[i][0], train_prefix[i][1])
                for i in range(overlap))
    return total / overlap


def knn_predict(query_prefix: List[Tuple[float, float]],
                pred_len: int,
                index: List[Tuple[List, List]],  # (prefix, continuation) pairs
                k: int = 5) -> List[Tuple[float, float]]:
    """Find k nearest neighbours and average their continuations."""
    dists = []
    for train_prefix, train_cont in index:
        d = prefix_dist(query_prefix, train_prefix)
        dists.append((d, train_cont))

    dists.sort(key=lambda x: x[0])
    neighbors = dists[:k]

    predicted = []
    for step in range(pred_len):
        lats = [n[1][step][0] for n in neighbors if len(n[1]) > step]
        lons = [n[1][step][1] for n in neighbors if len(n[1]) > step]
        if lats:
            predicted.append((float(np.mean(lats)), float(np.mean(lons))))
        elif predicted:
            predicted.append(predicted[-1])   # extrapolate last known
        else:
            predicted.append(query_prefix[-1])  # fall back to last prefix point
    return predicted


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(train: List[dict], test: List[dict], k: int = 5) -> dict:
    """Evaluate KNN on all test flights. Returns per-phase MHE lists."""
    # Build per-phase training index: {phase_label: [(prefix, continuation), ...]}
    train_index: Dict[str, List] = defaultdict(list)
    for e in train:
        n = e["n"]
        split = n // 2
        prefix = e["latlon"][:split]
        cont = e["latlon"][split:]
        train_index[e["phase_label"]].append((prefix, cont))

    phase_errors: Dict[str, List[float]] = defaultdict(list)
    skipped = 0

    for e in test:
        ph = e["phase_label"]
        if not train_index[ph]:
            skipped += 1
            continue
        n = e["n"]
        split = n // 2
        prefix = e["latlon"][:split]
        ground_truth = e["latlon"][split:]
        pred_len = len(ground_truth)
        if pred_len == 0:
            continue

        predicted = knn_predict(prefix, pred_len, train_index[ph], k=k)

        errors = [haversine_km(ground_truth[i][0], ground_truth[i][1],
                               predicted[i][0], predicted[i][1])
                  for i in range(min(len(ground_truth), len(predicted)))]
        if errors:
            phase_errors[ph].append(float(np.mean(errors)))

    if skipped:
        print(f"[WARN] Skipped {skipped} test entries with empty training index.")

    return dict(phase_errors)


def summarise(phase_errors: Dict[str, List[float]]) -> dict:
    phase_names = ["takeoff", "cruise", "landing"]
    result = {}
    all_vals = []
    for ph in phase_names:
        vals = phase_errors.get(ph, [])
        result[ph] = float(np.mean(vals)) if vals else float("nan")
        all_vals.extend(vals)
    result["overall"] = float(np.mean(all_vals)) if all_vals else float("nan")
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Physical KNN trajectory baseline")
    ap.add_argument("--data_root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--k", type=int, default=5, help="Number of neighbours")
    ap.add_argument("--runs", type=int, default=3,
                    help="Repeat runs to confirm std=0 (KNN is deterministic)")
    ap.add_argument("--train_frac", type=float, default=0.83)
    args = ap.parse_args()

    print(f"Loading data from: {os.path.abspath(args.data_root)}")
    entries = load_all_phases(args.data_root)
    print(f"Phase-level trajectories: {len(entries)}")

    train, test = chronological_split(entries, args.train_frac)
    print(f"Train: {len(train)} entries  |  Test: {len(test)} entries")

    # Phase breakdown
    from collections import Counter
    tc = Counter(e["phase_label"] for e in train)
    vc = Counter(e["phase_label"] for e in test)
    for ph in ["takeoff", "cruise", "landing"]:
        print(f"  {ph:<10s}: train={tc[ph]}, test={vc[ph]}")

    # Run multiple times (KNN is deterministic → confirms std=0)
    all_results = []
    for run in range(args.runs):
        print(f"\n--- Run {run + 1}/{args.runs} ---")
        phase_errors = evaluate(train, test, k=args.k)
        res = summarise(phase_errors)
        all_results.append(res)
        _print_result("KNN", res)

    # Aggregate
    phase_names = ["takeoff", "cruise", "landing", "overall"]
    agg = {}
    for ph in phase_names:
        vals = [r[ph] for r in all_results if not math.isnan(r.get(ph, float("nan")))]
        agg[ph] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "runs": vals,
        }

    print(f"\n{'='*54}")
    print(f"  KNN (k={args.k})  —  Mean Haversine Error (km)  [{args.runs} runs]")
    print(f"{'='*54}")
    for ph in phase_names:
        m, s = agg[ph]["mean"], agg[ph]["std"]
        print(f"  {ph:<10s}: {m:7.3f} ± {s:.3f}")
    print(f"{'='*54}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"model": f"KNN_k{args.k}", "phases": agg,
           "k": args.k, "runs": args.runs, "train_frac": args.train_frac}
    fpath = os.path.join(RESULTS_DIR, "KNN_results.json")
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {fpath}")


def _print_result(name, res):
    for ph in ["takeoff", "cruise", "landing", "overall"]:
        print(f"  {ph:<10s}: {res.get(ph, float('nan')):8.3f} km")


if __name__ == "__main__":
    main()
