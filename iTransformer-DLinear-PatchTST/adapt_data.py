"""
Convert TrajRAG phase JSON files to the flat CSV format expected by run_baseline.py.

Reads flight_data_with_minutes_since_start.json from all 6 route folders,
sorts all flights chronologically, assigns sequential integer flight_ids
(each phase of each real flight gets its own ID), and writes adapted_flights.csv.

Column mapping:
  fr24_id + phase          → flight_id  (sequential integer, chronological order)
  step index (0-based)     → t
  lat / lon                → lat / lon
  altitude_ft              → baro_alt
  gspeed                   → gspeed
  track                    → heading
  wind_speed + wind_dir    → wind_u, wind_v  (meteorological decomposition)
  edr_300hPa_m23s1         → turb

Usage:
    python adapt_data.py
    python adapt_data.py --data_root ../data --out adapted_flights.csv
"""
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd

ROUTES = ["MRS-MUC", "NTE-MUC", "BCN-MUC", "BCN-CDG", "CDG-FCO", "BOD-VCE"]
PHASES = ["take_off", "cruising", "landing"]
PHASE_LABEL = {"take_off": "takeoff", "cruising": "cruise", "landing": "landing"}
MIN_STEPS = 4  # drop phases with fewer steps than this


def wind_uv(speed_ms: float, dir_deg: float):
    """Meteorological wind → (u, v): u = eastward, v = northward component."""
    r = math.radians(dir_deg)
    return speed_ms * math.sin(r), speed_ms * math.cos(r)


def parse_ts(ts_str: str) -> datetime:
    # handles "2025-04-01 04:11:59+00:00" and "2025-04-01T04:11:59Z"
    for fmt in ("%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str[:19], fmt[:len(fmt.replace("+00:00","").replace("Z",""))])
        except Exception:
            pass
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def load_all_phases(data_root: str):
    """Return list of dicts: {fr24_id, route, phase, first_ts, points}."""
    entries = []
    for route in ROUTES:
        path = os.path.join(data_root, route, "flight_data_with_minutes_since_start.json")
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
                entries.append({
                    "fr24_id": fr24_id,
                    "route": route,
                    "phase": phase,
                    "first_ts": first_ts,
                    "points": pts_sorted,
                })
    return entries


def build_csv_rows(entries):
    """
    Sort entries chronologically (by real-flight first timestamp, then phase order),
    assign sequential integer flight_ids, flatten to rows.
    """
    # Group by real flight (fr24_id) to keep phases of same flight together
    by_flight = defaultdict(list)
    for e in entries:
        by_flight[e["fr24_id"]].append(e)

    # Sort real flights by the earliest timestamp of any their phases
    flight_order = sorted(
        by_flight.keys(),
        key=lambda fid: min(e["first_ts"] for e in by_flight[fid])
    )

    phase_rank = {p: i for i, p in enumerate(PHASES)}
    rows = []
    virtual_id = 0

    for fr24_id in flight_order:
        phases_for_flight = sorted(by_flight[fr24_id],
                                   key=lambda e: phase_rank.get(e["phase"], 99))
        for entry in phases_for_flight:
            pts = entry["points"]
            for t_idx, pt in enumerate(pts):
                spd = float(pt.get("wind_speed_10000m_ms", 0.0) or 0.0)
                drn = float(pt.get("wind_dir_10000m_d", 0.0) or 0.0)
                wu, wv = wind_uv(spd, drn)
                rows.append({
                    "flight_id": virtual_id,
                    "t": t_idx,
                    "lat": float(pt.get("lat", 0.0)),
                    "lon": float(pt.get("lon", 0.0)),
                    "baro_alt": float(pt.get("altitude_ft", 0.0) or 0.0),
                    "gspeed": float(pt.get("gspeed", 0.0) or 0.0),
                    "heading": float(pt.get("track", 0.0) or 0.0),
                    "wind_u": wu,
                    "wind_v": wv,
                    "turb": float(pt.get("edr_300hPa_m23s1", 0.0) or 0.0),
                    # metadata (not used by harness but useful for debugging)
                    "phase": PHASE_LABEL[entry["phase"]],
                    "fr24_id": fr24_id,
                    "route": entry["route"],
                })
            virtual_id += 1

    return rows, virtual_id


def main():
    ap = argparse.ArgumentParser(description="Convert TrajRAG JSON to TSLib CSV")
    ap.add_argument("--data_root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "adapted_flights.csv"))
    ap.add_argument("--min_steps", type=int, default=MIN_STEPS)
    args = ap.parse_args()

    print(f"Loading data from: {os.path.abspath(args.data_root)}")
    entries = load_all_phases(args.data_root)
    print(f"Phase-level trajectories loaded: {len(entries)}")

    rows, n_virtual = build_csv_rows(entries)
    df = pd.DataFrame(rows)

    # Fill any remaining NaN with 0
    numeric_cols = ["lat", "lon", "baro_alt", "gspeed", "heading",
                    "wind_u", "wind_v", "turb"]
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Summary
    steps = df.groupby("flight_id")["t"].count()
    phase_counts = df.drop_duplicates("flight_id")["phase"].value_counts()
    route_counts = df.drop_duplicates("flight_id")["route"].value_counts()

    print(f"\n{'='*50}")
    print(f"  Total virtual flights (phase-level): {n_virtual}")
    print(f"  Total rows:                          {len(df)}")
    print(f"  Steps per flight — min={steps.min()}  max={steps.max()}  "
          f"mean={steps.mean():.1f}  median={steps.median():.0f}")
    print(f"\n  Phase distribution:")
    for ph, cnt in phase_counts.items():
        print(f"    {ph:<12s}: {cnt}")
    print(f"\n  Route distribution:")
    for rt, cnt in route_counts.items():
        print(f"    {rt:<12s}: {cnt}")
    print(f"{'='*50}")

    df.to_csv(args.out, index=False)
    print(f"\nOutput written to: {os.path.abspath(args.out)}")

    # Cross-check: 83/17 split
    ids = sorted(df["flight_id"].unique())
    cut = int(len(ids) * 0.83)
    print(f"Chronological split (83/17): "
          f"train={cut} virtual flights, test={len(ids)-cut} virtual flights")


if __name__ == "__main__":
    main()
