"""
Train a TSLib model as a GNSS-blackout trajectory reconstructor.

Paper protocol (ICDM ADS 2026): seq_len=5, pred_len=5, epochs=30 by default.
  - Validation split (15% of train flights) for best-checkpoint selection
  - ReduceLROnPlateau scheduler (patience=7)
  - Weight decay in Adam; gradient clipping
  - Runs full epoch budget; restores best validation checkpoint
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import Configs
from data import (
    FEATURES, TARGET_IDX, PHASES, Scaler,
    TrajReconstructionDataset, haversine_km, make_synthetic_csv,
)
from models_wrapper import build_model

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.83):
    ids = sorted(df["flight_id"].unique())
    cut = int(len(ids) * train_frac)
    train_ids = set(ids[:cut])
    test_ids = set(ids[cut:])
    return (
        df[df.flight_id.isin(train_ids)],
        df[df.flight_id.isin(test_ids)],
    )


def val_split(train_df: pd.DataFrame, val_frac: float = 0.15):
    """Reserve the last val_frac of training flights for validation."""
    ids = sorted(train_df["flight_id"].unique())
    cut = int(len(ids) * (1 - val_frac))
    tr_ids = set(ids[:cut])
    vl_ids = set(ids[cut:])
    return (
        train_df[train_df.flight_id.isin(tr_ids)],
        train_df[train_df.flight_id.isin(vl_ids)],
    )


def train_one(model, tr_loader, vl_loader, device,
              epochs, lr, weight_decay=1e-4):
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        # train
        model.train()
        tr_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, None, None, None)
            loss = loss_fn(
                out[:, :, TARGET_IDX],
                y[:, :, TARGET_IDX],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
            n += x.size(0)

        # validate (for LR scheduler and best-model tracking only)
        model.eval()
        vl_loss, vn = 0.0, 0
        with torch.no_grad():
            for x, y, _ in vl_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, None, None, None)
                vl_loss += loss_fn(
                    out[:, :, TARGET_IDX],
                    y[:, :, TARGET_IDX],
                ).item() * x.size(0)
                vn += x.size(0)

        vl_avg = vl_loss / max(vn, 1)
        scheduler.step(vl_avg)

        if (ep + 1) % 10 == 0 or ep == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"  ep {ep+1:03d}/{epochs}"
                f"  tr={tr_loss/max(n,1):.4f}"
                f"  vl={vl_avg:.4f}"
                f"  lr={lr_now:.2e}"
            )

        if vl_avg < best_val - 1e-6:
            best_val = vl_avg
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, device, scaler):
    model.eval()
    errs = {p: [] for p in PHASES}
    all_errs = []
    for x, y, phases in loader:
        x = x.to(device)
        out = (
            model(x, None, None, None)
            [:, :, TARGET_IDX].cpu().numpy()
        )
        true = y[:, :, TARGET_IDX].numpy()
        pred_ll = scaler.inverse_targets(out)
        true_ll = scaler.inverse_targets(true)
        d = haversine_km(
            true_ll[..., 0], true_ll[..., 1],
            pred_ll[..., 0], pred_ll[..., 1],
        )
        for e, ph in zip(d.mean(axis=1), phases):
            errs.get(ph, all_errs).append(float(e))
            all_errs.append(float(e))
    summary = {
        p: float(np.mean(v)) if v else float("nan")
        for p, v in errs.items()
    }
    summary["overall"] = (
        float(np.mean(all_errs)) if all_errs else float("nan")
    )
    return summary


def make_configs(args, n_features):
    return Configs(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
    )


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="iTransformer",
        choices=["iTransformer", "PatchTST", "DLinear"],
    )
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seq_len", type=int, default=5)
    ap.add_argument("--pred_len", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.83)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device}  model={args.model}"
        f"  seq={args.seq_len}  pred={args.pred_len}"
        f"  seed={args.seed}"
    )

    csv = args.csv or make_synthetic_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "synthetic_flights.csv",
        ),
        seed=args.seed,
    )
    df = pd.read_csv(csv)
    train_df, test_df = chronological_split(df, args.train_frac)
    sub_train_df, vl_df = val_split(train_df)

    scaler = Scaler().fit(
        sub_train_df[FEATURES].to_numpy(dtype=float)
    )
    n_feats = len(FEATURES)

    tr = TrajReconstructionDataset(
        sub_train_df, args.seq_len, args.pred_len, scaler
    )
    vl = TrajReconstructionDataset(
        vl_df, args.seq_len, args.pred_len, scaler
    )
    te = TrajReconstructionDataset(
        test_df, args.seq_len, args.pred_len, scaler
    )

    print(
        f"windows — train={len(tr)}"
        f"  val={len(vl)}  test={len(te)}"
    )
    if len(tr) == 0 or len(te) == 0:
        print("[ERROR] No windows. Check seq_len vs flight lengths.")
        return None

    model = build_model(
        args.model, make_configs(args, n_feats)
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    train_one(
        model,
        make_loader(tr, args.batch_size, shuffle=True),
        make_loader(vl, args.batch_size),
        device, args.epochs, args.lr,
    )

    res = evaluate(
        model,
        make_loader(te, args.batch_size),
        device, scaler,
    )
    report(args.model, res, args)
    return res


def report(name, res, args=None):
    print("\n" + "=" * 50)
    print(f"  {name}  —  Mean Haversine Error (km)")
    print("=" * 50)
    for p in PHASES:
        print(f"  {p:<10s}: {res.get(p, float('nan')):8.3f}")
    print("-" * 50)
    print(
        f"  {'overall':<10s}:"
        f" {res.get('overall', float('nan')):8.3f}"
    )
    print("=" * 50)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seed = getattr(args, "seed", 0) if args else 0
    out = {
        p: res.get(p, float("nan"))
        for p in PHASES + ["overall"]
    }
    fpath = os.path.join(RESULTS_DIR, f"{name}_seed{seed}.json")
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {fpath}")


if __name__ == "__main__":
    main()
