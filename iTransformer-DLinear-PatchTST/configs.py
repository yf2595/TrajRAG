"""
Config object for TSLib models (iTransformer / PatchTST / DLinear).

Paper protocol (ICDM ADS 2026): seq_len=5, pred_len=5, epochs=30.
Disclose: fixed pred_len vs TrajRAG variable blackout window (see RUN_EXPERIMENTS.md).
"""
from dataclasses import dataclass


@dataclass
class Configs:
    # --- task / shapes ---
    task_name: str = "long_term_forecast"
    seq_len: int = 5      # paper protocol; takeoff phases <10 steps may be skipped
    label_len: int = 0
    pred_len: int = 5
    enc_in: int = 8
    dec_in: int = 8
    c_out: int = 8

    # --- model capacity ---
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"

    # --- embedding ---
    embed: str = "timeF"
    freq: str = "t"

    # --- model-specific ---
    moving_avg: int = 3
    output_attention: bool = False
    num_class: int = 0

    # PatchTST
    patch_len: int = 2
    stride: int = 1

    def for_model(self, name: str) -> "Configs":
        if name == "DLinear":
            k = min(self.moving_avg, self.seq_len)
            if k % 2 == 0:
                k -= 1
            self.moving_avg = max(k, 3)
        if name == "PatchTST":
            self.patch_len = min(self.patch_len, self.seq_len)
            self.stride = max(1, self.patch_len // 2)
        return self
