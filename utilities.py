import tensorflow as tf
import matplotlib.pyplot as plt
import gc
def soft_reset(tag=""):
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        plt.close("all")
    except Exception:
        pass
    for _ in range(3):
        gc.collect()
    if tag:
        print(f"[soft_reset] done: {tag}")


import numpy as np

def take_reg_labels(dts: dict, col: str = "error ratio") -> dict:
    out = {}
    for k, df in dts.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in split '{k}'.")
        out[k] = df[col].to_numpy(dtype=np.float32)
    return out


def take_domains_local(balanced_dtsets: dict, s_d: dict) -> dict:
    """
    Domain ids:
      TRAIN1..TRAINk => 0..k-1
      ADAPTION/test_adaption/TEST => k (PL_DOMAIN_ID)
    """
    k = int(s_d["PL_DOMAIN_ID"])
    out = {}

    # sources
    for i in range(len(s_d["Training_labels"])):
        key = f"TRAIN{i+1}"
        n = len(balanced_dtsets[key])
        out[key] = np.full((n,), i, dtype=np.int32)

    # target splits
    for key in ["ADAPTION", "test_adaption", "TEST"]:
        n = len(balanced_dtsets[key])
        out[key] = np.full((n,), k, dtype=np.int32)

    return out


def take_weights_local(balanced_dtsets: dict, s_d: dict) -> dict:
    """
    Task weights:
      TRAIN* => 1
      ADAPTION => 0 if UDA else 1
      test_adaption/TEST => 1 (evaluation)
    """
    out = {}
    uda = int(s_d["ADAPTION_WITH_LABEL"]) == 0

    for i in range(len(s_d["Training_labels"])):
        key = f"TRAIN{i+1}"
        n = len(balanced_dtsets[key])
        out[key] = np.ones((n,), dtype=np.float32)

    n_ad = len(balanced_dtsets["ADAPTION"])
    out["ADAPTION"] = np.zeros((n_ad,), dtype=np.float32) if uda else np.ones((n_ad,), dtype=np.float32)

    for key in ["test_adaption", "TEST"]:
        n = len(balanced_dtsets[key])
        out[key] = np.ones((n,), dtype=np.float32)

    return out

import os
import json
from pathlib import Path

def get_coral_label_mode(default: str = "ratio") -> str:
    """
    Single source of truth for label mode.
    Reads env CORAL_LABEL_MODE and validates it.
    """
    mode = os.environ.get("CORAL_LABEL_MODE", default).strip().lower()
    if mode not in ("ratio", "error"):
        raise ValueError(f"Invalid CORAL_LABEL_MODE='{mode}'. Use 'ratio' or 'error'.")
    return mode


def save_run_meta(save_dir, **extra):
    """Save metadata next to model outputs."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "CORAL_LABEL_MODE": get_coral_label_mode(),
        **extra
    }
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_run_meta(model_dir):
    """Load metadata if exists; return dict or {}."""
    model_dir = Path(model_dir)
    meta_path = model_dir / "run_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

