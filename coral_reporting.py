# coral_reporting.py
# ============================================================
# Save evaluation outputs to Excel + plots for CORAL regression
# Target use-case:
#   y = error_ratio
#   error = y * sensor_rng
#   pred_rng = sensor_rng - pred_error
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union
from typing import Literal, Optional
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import tensorflow as tf


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _load_model_if_needed(model_or_path):
    if isinstance(model_or_path, (str, Path)):
        return tf.keras.models.load_model(str(model_or_path), compile=False)
    return model_or_path


def _predict_ratio(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    out = model.predict(X, batch_size=batch_size, verbose=0)
    # expected: dict {"pred":..., "latent":...} OR tuple/list OR array
    if isinstance(out, dict):
        y_pred = out.get("pred", None)
        if y_pred is None:
            # fallback: first value
            y_pred = list(out.values())[0]
    elif isinstance(out, (tuple, list)):
        y_pred = out[0]
    else:
        y_pred = out
    y_pred = _as_1d(np.asarray(y_pred, dtype=np.float32))
    return y_pred

# +++ rename to reflect it may be ratio OR error (generic regression head)
def _predict_reg(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return _predict_ratio(model, X, batch_size=batch_size)


def _mae(y_true, y_pred) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def _mse(y_true, y_pred) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    e = (y_true - y_pred)
    return float(np.mean(e * e))


# Set globally here:
#   "ratio" -> model outputs error_ratio
#   "error" -> model outputs raw error (meters)
DEFAULT_LABEL_MODE: Literal["ratio", "error"] = "ratio"

def _global_label_mode() -> Literal["ratio", "error"]:
    v = os.getenv("CORAL_LABEL_MODE", DEFAULT_LABEL_MODE)
    v = (v or DEFAULT_LABEL_MODE).strip().lower()
    return "error" if v == "error" else "ratio"

def _resolve_label_mode(label_mode: Optional[Literal["ratio", "error"]]) -> Literal["ratio", "error"]:
    return _global_label_mode() if label_mode is None else label_mode


def save_eval_excel_and_plots(
    save_dir: Union[str, Path],
    model_or_path,
    splits: Dict[str, Dict[str, Any]],
    batch_size: int = 256,
    excel_name: str = "eval_report.xlsx",
    max_plot_points: int = 5000,
    series_points: int = 300,
    label_mode: Optional[Literal["ratio", "error"]] = None,
    y_key: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    splits format (per split):
      {
        "X": np.ndarray [N,T,1],
        # if label_mode == "ratio": provide "y_ratio" (or set y_key)
        # if label_mode == "error": provide "y_error" (or set y_key)
        "sensor_rng": np.ndarray [N],
        "camera_rng": np.ndarray [N],
        "domain_id": (optional) np.ndarray [N],
      }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model_if_needed(model_or_path)

    # +++ resolve once so the rest of the function is consistent
    label_mode = _resolve_label_mode(label_mode)

    metrics = {}
    all_rows = []

    xlsx_path = save_dir / excel_name
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for split_name, pack in splits.items():
            X = pack["X"]

            # +++ robust label loading (avoid _as_1d(None) crashes)
            if y_key is not None:
                if y_key not in pack:
                    raise KeyError(f"y_key='{y_key}' not found for split '{split_name}'. Available: {list(pack.keys())}")
                y_label = _as_1d(pack[y_key]).astype(np.float32)
            else:
                if label_mode == "ratio":
                    if "y_ratio" in pack:
                        y_label = _as_1d(pack["y_ratio"]).astype(np.float32)
                    elif "y" in pack:
                        y_label = _as_1d(pack["y"]).astype(np.float32)
                    else:
                        raise KeyError(f"Missing y_ratio (or y) for split '{split_name}'.")
                else:
                    if "y_error" in pack:
                        y_label = _as_1d(pack["y_error"]).astype(np.float32)
                    elif "y" in pack:
                        y_label = _as_1d(pack["y"]).astype(np.float32)
                    else:
                        raise KeyError(f"Missing y_error (or y) for split '{split_name}'.")

            sensor_rng = _as_1d(pack["sensor_rng"]).astype(np.float32)
            camera_rng = _as_1d(pack["camera_rng"]).astype(np.float32)

            y_pred_label = _predict_reg(model, X, batch_size=batch_size)

            eps = 1e-6

            if label_mode == "ratio":
                real_error = y_label * sensor_rng
                pred_error = y_pred_label * sensor_rng
            else:
                real_error = y_label
                pred_error = y_pred_label

            y_true_ratio = real_error / (sensor_rng + eps)
            y_pred_ratio = pred_error / (sensor_rng + eps)

            pred_rng = sensor_rng - pred_error

            mae_ratio = _mae(y_true_ratio, y_pred_ratio)
            mse_ratio = _mse(y_true_ratio, y_pred_ratio)

            mae_error = _mae(real_error, pred_error)
            mse_error = _mse(real_error, pred_error)

            mae_rng = _mae(camera_rng, pred_rng)
            mse_rng = _mse(camera_rng, pred_rng)

            metrics[split_name] = {
                "mae_ratio": mae_ratio,
                "mse_ratio": mse_ratio,
                "mae_error": mae_error,
                "mse_error": mse_error,
                "mae_rng": mae_rng,
                "mse_rng": mse_rng,
                "n": int(len(sensor_rng)),
            }
            all_rows.append({"split": split_name, **metrics[split_name]})

            df = pd.DataFrame({
                "label_mode": label_mode,
                "y_true_label": y_label,
                "y_pred_label": y_pred_label,
                "y_true_ratio": y_true_ratio,
                "y_pred_ratio": y_pred_ratio,
                "sensor_rng": sensor_rng,
                "camera_rng": camera_rng,
                "real_error": real_error,
                "pred_error": pred_error,
                "pred_rng": pred_rng,
                "ratio_abs_err": np.abs(y_true_ratio - y_pred_ratio),
                "ratio_sq_err": (y_true_ratio - y_pred_ratio) ** 2,
                "rng_abs_err": np.abs(camera_rng - pred_rng),
                "rng_sq_err": (camera_rng - pred_rng) ** 2,
            })

            if "domain_id" in pack and pack["domain_id"] is not None:
                df["domain_id"] = _as_1d(pack["domain_id"]).astype(np.int32)

            sheet = split_name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            df.to_csv(save_dir / f"eval_{split_name}.csv", index=False)

            # +++ ensure plot_dir exists (it was missing in your current file)
            plot_dir = save_dir / f"plots_{split_name}"
            plot_dir.mkdir(parents=True, exist_ok=True)

            n = len(sensor_rng)
            idx = np.arange(n)
            if n > max_plot_points:
                idx = idx[:max_plot_points]

            yt = y_true_ratio[idx]
            yp = y_pred_ratio[idx]
            cr = camera_rng[idx]
            pr = pred_rng[idx]

            # 1) scatter ratio: true vs pred
            plt.figure()
            plt.scatter(yt, yp, s=4)
            plt.xlabel("True error ratio")
            plt.ylabel("Pred error ratio")
            plt.title(f"{split_name} | ratio: true vs pred ({label_mode})")
            plt.tight_layout()
            plt.savefig(plot_dir / "scatter_ratio_true_vs_pred.png", dpi=200)
            plt.close()

            # 2) scatter range: camera vs pred_rng
            plt.figure()
            plt.scatter(cr, pr, s=4)
            plt.xlabel("Camera rng (m)")
            plt.ylabel("Predicted corrected rng (m)")
            plt.title(f"{split_name} | range: camera vs predicted")
            plt.tight_layout()
            plt.savefig(plot_dir / "scatter_range_camera_vs_pred.png", dpi=200)
            plt.close()

            # 3) histogram residuals on ratio
            plt.figure()
            plt.hist((yt - yp), bins=60)
            plt.xlabel("Residual (true_ratio - pred_ratio)")
            plt.ylabel("Count")
            plt.title(f"{split_name} | ratio residual histogram")
            plt.tight_layout()
            plt.savefig(plot_dir / "hist_ratio_residual.png", dpi=200)
            plt.close()

            # 4) line series (first K samples)
            K = min(series_points, n)
            plt.figure()
            plt.plot(sensor_rng[:K], label="Sensor rng")
            plt.plot(camera_rng[:K], label="Camera rng")
            plt.plot(pred_rng[:K], label="Pred corrected rng")
            plt.xlabel("sample index")
            plt.ylabel("range (m)")
            plt.title(f"{split_name} | ranges (first {K})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "series_ranges_firstK.png", dpi=200)
            plt.close()

        # summary sheet
        df_sum = pd.DataFrame(all_rows)
        df_sum.to_excel(writer, sheet_name="summary", index=False)

    # also save summary csv
    pd.DataFrame(all_rows).to_csv(save_dir / "eval_summary.csv", index=False)
    return metrics


def save_range_metrics_excel(
    save_dir,
    model_or_path,
    splits,
    batch_size: int = 256,
    excel_name: str = "range_metrics.xlsx",
    label_mode: Optional[Literal["ratio", "error"]] = None,
):
    """
    Saves a separate Excel file with range errors (in meters):
      - camera vs sensor
      - camera vs predicted_rng (derived from predicted label)

    label_mode:
      - "ratio": model output interpreted as error_ratio, pred_error = y_pred * sensor_rng
      - "error": model output interpreted as raw error (m), pred_error = y_pred
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model_if_needed(model_or_path)

    label_mode = _resolve_label_mode(label_mode)

    rows = []
    per_split_tables = {}

    for split_name, pack in splits.items():
        X = pack["X"]
        sensor_rng = _as_1d(pack["sensor_rng"]).astype(np.float32)
        camera_rng = _as_1d(pack["camera_rng"]).astype(np.float32)

        # baseline (sensor only)
        base_err = camera_rng - sensor_rng
        base_mae = float(np.mean(np.abs(base_err)))
        base_mse = float(np.mean(base_err * base_err))

        # model predicted label -> predicted corrected range
        y_pred_label = _predict_reg(model, X, batch_size=batch_size)
        if label_mode == "ratio":
            pred_error = y_pred_label * sensor_rng
        else:
            pred_error = y_pred_label
        pred_rng = sensor_rng - pred_error

        pred_err = camera_rng - pred_rng
        pred_mae = float(np.mean(np.abs(pred_err)))
        pred_mse = float(np.mean(pred_err * pred_err))

        improvement_mae_pct = 100.0 * (base_mae - pred_mae) / (base_mae + 1e-12)
        improvement_mse_pct = 100.0 * (base_mse - pred_mse) / (base_mse + 1e-12)

        rows.append({
            "split": split_name,
            "N": int(len(camera_rng)),
            "MAE(camera, sensor)": base_mae,
            "MSE(camera, sensor)": base_mse,
            "MAE(camera, predicted_rng)": pred_mae,
            "MSE(camera, predicted_rng)": pred_mse,
            "MAE_improvement_%": improvement_mae_pct,
            "MSE_improvement_%": improvement_mse_pct,
        })

        # optional per-sample table (handy for debugging)
        per_split_tables[split_name] = pd.DataFrame({
            "camera_rng": camera_rng,
            "sensor_rng": sensor_rng,
            "predicted_rng": pred_rng,
            "baseline_err(camera-sensor)": base_err,
            "pred_err(camera-pred)": pred_err,
        })

    df_summary = pd.DataFrame(rows)

    xlsx_path = save_dir / excel_name
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)

        # also include per-split sheets (you can remove this if you want only summary)
        for split_name, df_split in per_split_tables.items():
            sheet = split_name[:31]
            df_split.to_excel(writer, sheet_name=sheet, index=False)

    # also save csv summary
    df_summary.to_csv(save_dir / "range_metrics_summary.csv", index=False)

    return df_summary
