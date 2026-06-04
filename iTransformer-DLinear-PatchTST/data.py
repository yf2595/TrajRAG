"""
Trajectory data for blackout reconstruction.

Expected input CSV columns (one row per ADS-B state):
    flight_id, t, lat, lon, baro_alt, gspeed, heading, wind_u, wind_v, turb

`t` is a step index (monotonic within a flight, 0-based).

Modelling decisions matched to TrajRAG protocol:
  * PREFIX (encoder input, length seq_len)  = trusted pre-detection segment.
  * HORIZON (length pred_len)               = GNSS blackout. Model forecasts
    lat/lon there — never sees corrupted positions. Direct multi-step reconstruction.
  * Only lat & lon are scored (Haversine). Other channels are context only.
  * Each window is tagged with a flight phase (takeoff/cruise/landing) inferred
    from barometric-altitude trend — GNSS-independent, as in the paper.
"""
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ["lat", "lon", "baro_alt", "gspeed", "heading", "wind_u", "wind_v", "turb"]
TARGETS = ["lat", "lon"]
TARGET_IDX = [FEATURES.index(c) for c in TARGETS]
PHASES = ["takeoff", "cruise", "landing"]


# ----------------------------------------------------------------------------- #
# Phase segmentation — altitude-based, GNSS-independent
# ----------------------------------------------------------------------------- #
def infer_phase_per_step(baro_alt: np.ndarray) -> np.ndarray:
    alt = np.asarray(baro_alt, dtype=float)
    n = len(alt)
    if n < 3:
        return np.array(["cruise"] * n)
    cruise_thr = 0.80 * np.nanmax(alt)
    slope = np.gradient(alt)
    labels = []
    for i in range(n):
        if alt[i] < cruise_thr and slope[i] > 0 and i < n / 2:
            labels.append("takeoff")
        elif alt[i] < cruise_thr and slope[i] < 0 and i > n / 2:
            labels.append("landing")
        else:
            labels.append("cruise")
    return np.array(labels)


# ----------------------------------------------------------------------------- #
# Per-channel standardization (fit on TRAIN only)
# ----------------------------------------------------------------------------- #
class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_targets(self, y_std: np.ndarray) -> np.ndarray:
        m = self.mean[TARGET_IDX]
        s = self.std[TARGET_IDX]
        return y_std * s + m


# ----------------------------------------------------------------------------- #
# Windowed dataset
# ----------------------------------------------------------------------------- #
class TrajReconstructionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int,
                 scaler: Scaler, phase: Optional[str] = None, stride: int = 1):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.windows = []
        for _, g in df.sort_values(["flight_id", "t"]).groupby("flight_id"):
            arr = g[FEATURES].to_numpy(dtype=float)
            step_phase = infer_phase_per_step(g["baro_alt"].to_numpy())
            arr = scaler.transform(arr)
            need = seq_len + pred_len
            for s in range(0, len(arr) - need + 1, stride):
                x = arr[s:s + seq_len]
                y = arr[s + seq_len:s + seq_len + pred_len]
                ph = step_phase[s + seq_len - 1]
                if phase is None or ph == phase:
                    self.windows.append((x.astype(np.float32),
                                         y.astype(np.float32), ph))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        x, y, ph = self.windows[i]
        return torch.from_numpy(x), torch.from_numpy(y), ph


# ----------------------------------------------------------------------------- #
# Metric: mean Haversine error (km)
# ----------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ----------------------------------------------------------------------------- #
# Synthetic data generator (smoke test)
# ----------------------------------------------------------------------------- #
def make_synthetic_csv(path: str, n_flights: int = 200, n_steps: int = 80,
                       seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    A = np.array([48.85, 2.35])
    B = np.array([41.90, 12.50])
    for fid in range(n_flights):
        jitter = rng.normal(0, 0.05, size=2)
        ctrl = (A + B) / 2 + rng.normal(0, 0.4, size=2)
        ts = np.linspace(0, 1, n_steps)
        pos = ((1 - ts)[:, None] ** 2 * (A + jitter)
               + 2 * (1 - ts)[:, None] * ts[:, None] * ctrl
               + ts[:, None] ** 2 * (B + jitter))
        lat, lon = pos[:, 0], pos[:, 1]
        lat += rng.normal(0, 0.01, n_steps)
        lon += rng.normal(0, 0.01, n_steps)
        alt = np.piecewise(
            ts, [ts < 0.30, (ts >= 0.30) & (ts < 0.70), ts >= 0.70],
            [lambda u: 11000 * (u / 0.30),
             lambda u: 11000 + rng.normal(0, 80, len(u)),
             lambda u: 11000 * (1 - (u - 0.70) / 0.30)])
        d = np.gradient(pos, axis=0)
        gspeed = np.linalg.norm(d, axis=1) * 6000 + rng.normal(0, 5, n_steps)
        heading = (np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360
        wind_u = rng.normal(5, 2, n_steps)
        wind_v = rng.normal(-3, 2, n_steps)
        turb = np.abs(rng.normal(0.2, 0.1, n_steps))
        for i in range(n_steps):
            rows.append(dict(flight_id=fid, t=i, lat=lat[i], lon=lon[i],
                             baro_alt=alt[i], gspeed=gspeed[i], heading=heading[i],
                             wind_u=wind_u[i], wind_v=wind_v[i], turb=turb[i]))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
