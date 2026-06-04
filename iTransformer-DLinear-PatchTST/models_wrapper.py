"""
Build a TSLib model by name. Requires the Time-Series-Library repo cloned locally:

    cd GPS-SPOOFING-CORRECTION
    git clone --depth 1 https://github.com/thuml/Time-Series-Library.git TSLib

TSLIB_PATH defaults to ./TSLib relative to this file (or set env var TSLIB_PATH).
"""
import os
import sys

TSLIB_PATH = os.environ.get(
    "TSLIB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSLib")
)
if TSLIB_PATH not in sys.path:
    sys.path.insert(0, TSLIB_PATH)

SUPPORTED = ["iTransformer", "PatchTST", "DLinear"]


def build_model(name: str, configs):
    configs.for_model(name)
    if name == "iTransformer":
        from models.iTransformer import Model
        return Model(configs)
    elif name == "PatchTST":
        from models.PatchTST import Model
        # TSLib PatchTST ignores configs.patch_len — must pass explicitly
        patch_len = min(configs.patch_len, configs.seq_len)
        stride = max(1, patch_len // 2)
        return Model(configs, patch_len=patch_len, stride=stride)
    elif name == "DLinear":
        from models.DLinear import Model
        return Model(configs)
    else:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED}")
