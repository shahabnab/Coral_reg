# training_engine.py
# =============================================================================
# Deep CORAL (multi-source -> target) for REGRESSION
# - No Python "if tf.rank(...) ..." anywhere (Graph-safe)
# - Reports MAE + MSE
# - Supports evaluating extra datasets (ADAPTION, test_adaption, ...)
#
# Input dataset element format expected:
#   (x_dict, y) where:
#     x_dict["x"] : float32 [B, T, 1]
#     x_dict["d"] : int32   [B] domain id
#     x_dict["w"] : float32 [B] sample weights
#
# Base model output must be either:
#   dict: {"pred": [B,1], "latent": [B,D]}
#   tuple: (pred, latent)
# =============================================================================

from __future__ import annotations
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────────────
# CORAL (Graph-safe)
# ──────────────────────────────────────────────────────────────────────────────

def coral_loss_safe(fs: tf.Tensor, ft: tf.Tensor) -> tf.Tensor:
    fs = tf.cast(fs, tf.float32)
    ft = tf.cast(ft, tf.float32)

    d = tf.cast(tf.shape(fs)[1], tf.float32)

    # mean
    ms = tf.reduce_mean(fs, axis=0, keepdims=True)
    mt = tf.reduce_mean(ft, axis=0, keepdims=True)

    fs_c = fs - ms
    ft_c = ft - mt

    # covariance (safe denom)
    ns = tf.cast(tf.shape(fs_c)[0], tf.float32)
    nt = tf.cast(tf.shape(ft_c)[0], tf.float32)

    Cs = tf.matmul(fs_c, fs_c, transpose_a=True) / tf.maximum(ns - 1.0, 1.0)
    Ct = tf.matmul(ft_c, ft_c, transpose_a=True) / tf.maximum(nt - 1.0, 1.0)

    return tf.reduce_sum(tf.square(Cs - Ct)) / (4.0 * d * d)

def _gather_domain(feats: tf.Tensor, dom_ids: tf.Tensor, dom_value: tf.Tensor) -> tf.Tensor:
    idx = tf.where(tf.equal(dom_ids, dom_value))[:, 0]
    return tf.gather(feats, idx)

def coral_multi_source_to_target(
    feats: tf.Tensor,
    dom_ids: tf.Tensor,
    target_id: int,
    source_ids: Tuple[int, ...],
) -> tf.Tensor:
    dom_ids = tf.cast(dom_ids, tf.int32)
    target_id = tf.cast(target_id, tf.int32)

    # ALWAYS reshape to [B, D] (Graph-safe)
    feats = tf.cast(feats, tf.float32)
    feats = tf.reshape(feats, [tf.shape(feats)[0], -1])

    ft = _gather_domain(feats, dom_ids, target_id)

    total = tf.constant(0.0, tf.float32)
    count = tf.constant(0.0, tf.float32)

    for sid in source_ids:
        sid_t = tf.cast(sid, tf.int32)
        fs = _gather_domain(feats, dom_ids, sid_t)

        ns = tf.shape(fs)[0]
        nt = tf.shape(ft)[0]
        valid = tf.cast(tf.logical_and(ns >= 2, nt >= 2), tf.float32)

        total = total + valid * coral_loss_safe(fs, ft)
        count = count + valid

    return total / tf.maximum(count, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Engine config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    task_type: str = "reg"
    adaption_with_label: int = 0

    use_coral: bool = True
    coral_lambda: Any = 0.2  # allow float OR tf.Variable (ramp-up)
    coral_target_id: int = 2
    coral_source_ids: Tuple[int, ...] = (0, 1)

    # NEW: reconstruction regularizer
    use_recon: bool = True
    recon_lambda: float = 0.0
    output_recon_key: str = "recon"

    grad_clipnorm: Optional[float] = 5.0
    output_pred_key: str = "pred"
    output_latent_key: str = "latent"


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_model_outputs(
    outputs: Union[Dict[str, tf.Tensor], Tuple[tf.Tensor, ...], List[tf.Tensor]],
    cfg: EngineConfig,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if isinstance(outputs, dict):
        if cfg.output_pred_key not in outputs:
            raise KeyError(f"Missing pred key '{cfg.output_pred_key}' in model outputs. Keys={list(outputs.keys())}")
        if cfg.output_latent_key not in outputs:
            raise KeyError(f"Missing latent key '{cfg.output_latent_key}' in model outputs. Keys={list(outputs.keys())}")
        return outputs[cfg.output_pred_key], outputs[cfg.output_latent_key]

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        return outputs[0], outputs[1]

    raise ValueError("Model outputs must be dict with pred/latent or tuple/list (pred, latent).")

def _weighted_mean(loss_vec: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    loss_vec = tf.cast(loss_vec, tf.float32)
    weights = tf.cast(weights, tf.float32)
    num = tf.reduce_sum(loss_vec * weights)
    den = tf.reduce_sum(weights)
    return tf.math.divide_no_nan(num, den)

def _mse_vec(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # reshape to [B, T] or [B, K]
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    # use common length
    L = tf.minimum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
    y_true = y_true[:, :L]
    y_pred = y_pred[:, :L]

    return tf.reduce_mean(tf.square(y_true - y_pred), axis=1)


def _lambda_tensor(x: Any) -> tf.Tensor:
    t = tf.convert_to_tensor(x, dtype=tf.float32)
    return tf.reshape(t, [])

def _lambda_float(x: Any) -> float:
    try:
        if isinstance(x, (tf.Variable, tf.Tensor)):
            return float(tf.reshape(tf.cast(x, tf.float32), []).numpy())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float("nan")

def _parse_recon(outputs: Union[Dict[str, tf.Tensor], Tuple[tf.Tensor, ...], List[tf.Tensor]], cfg: EngineConfig) -> Optional[tf.Tensor]:
    if isinstance(outputs, dict) and cfg.output_recon_key in outputs:
        return outputs[cfg.output_recon_key]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CoralTrainingModel(tf.keras.Model):
    def __init__(self, base: tf.keras.Model, cfg: EngineConfig, name="coral_engine"):
        super().__init__(name=name)
        self.base = base
        self.cfg = cfg

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.task_loss_tracker  = tf.keras.metrics.Mean(name="task_loss")
        self.coral_loss_tracker = tf.keras.metrics.Mean(name="coral_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")  # NEW

        self.mae = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.mse = tf.keras.metrics.MeanSquaredError(name="mse")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.task_loss_tracker,
            self.coral_loss_tracker,
            self.recon_loss_tracker,  # NEW
            self.mae,
            self.mse,
        ]

    def train_step(self, data):
        x_dict, y = data
        x = x_dict["x"]
        d = tf.cast(x_dict["d"], tf.int32)
        w = tf.cast(x_dict["w"], tf.float32)

        # labeled mask:
        src_ids = tf.constant(list(self.cfg.coral_source_ids), tf.int32)
        is_source = tf.reduce_any(tf.equal(tf.expand_dims(d, 1), tf.expand_dims(src_ids, 0)), axis=1)
        is_target = tf.equal(d, tf.cast(self.cfg.coral_target_id, tf.int32))
        allow_t   = tf.cast(self.cfg.adaption_with_label, tf.bool)
        is_labeled = tf.logical_or(is_source, tf.logical_and(is_target, allow_t))

        task_w = w * tf.cast(is_labeled, tf.float32)

        recon_w = tf.cast(self.cfg.recon_lambda, tf.float32)
        lam = _lambda_tensor(self.cfg.coral_lambda) if self.cfg.use_coral else tf.constant(0.0, tf.float32)

        with tf.GradientTape() as tape:
            outputs = self.base(x, training=True)
            y_pred, z = _parse_model_outputs(outputs, self.cfg)

            # task loss (labeled only)
            loss_vec = _mse_vec(y, y_pred)     # [B]
            task = _weighted_mean(loss_vec, task_w)

            # coral loss
            def _coral_branch():
                return coral_multi_source_to_target(
                    feats=z, dom_ids=d,
                    target_id=self.cfg.coral_target_id,
                    source_ids=self.cfg.coral_source_ids,
                )

            coral = tf.cond(tf.greater(lam, 0.0), _coral_branch, lambda: tf.constant(0.0, tf.float32))

            # recon loss (all samples)
            recon = tf.constant(0.0, tf.float32)
            if self.cfg.use_recon:
                x_hat = _parse_recon(outputs, self.cfg)
                if x_hat is not None:
                    recon_vec = _mse_vec(x, x_hat)   # [B]
                    recon = _weighted_mean(recon_vec, w)

            total = task + lam * coral + recon_w * recon

            opt = self.optimizer
            if hasattr(opt, "get_scaled_loss"):
                scaled_total = opt.get_scaled_loss(total)
            else:
                scaled_total = total

        grads = tape.gradient(scaled_total, self.base.trainable_variables)

        if hasattr(self.optimizer, "get_unscaled_gradients"):
            grads = self.optimizer.get_unscaled_gradients(grads)

        if self.cfg.grad_clipnorm is not None:
            grads = [tf.clip_by_norm(g, self.cfg.grad_clipnorm) if g is not None else None for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        # trackers
        self.total_loss_tracker.update_state(total)
        self.task_loss_tracker.update_state(task)
        self.coral_loss_tracker.update_state(coral)
        self.recon_loss_tracker.update_state(recon)

        # metrics on labeled samples only
        y_true_2d = tf.reshape(tf.cast(y, tf.float32), [tf.shape(y)[0], -1])
        y_pred_2d = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])

        self.mae.update_state(y_true_2d, y_pred_2d, sample_weight=task_w)
        self.mse.update_state(y_true_2d, y_pred_2d, sample_weight=task_w)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_dict, y = data
        x = x_dict["x"]
        w = tf.cast(x_dict["w"], tf.float32)

        outputs = self.base(x, training=False)
        y_pred, _ = _parse_model_outputs(outputs, self.cfg)

        loss_vec = _mse_vec(y, y_pred)
        task = _weighted_mean(loss_vec, w)

        recon = tf.constant(0.0, tf.float32)
        if self.cfg.use_recon:
            x_hat = _parse_recon(outputs, self.cfg)
            if x_hat is not None:
                recon_vec = _mse_vec(x, x_hat)
                recon = _weighted_mean(recon_vec, w)

        recon_w = tf.cast(self.cfg.recon_lambda, tf.float32)
        total = task + recon_w * recon

        self.total_loss_tracker.update_state(total)
        self.task_loss_tracker.update_state(task)
        self.coral_loss_tracker.update_state(0.0)
        self.recon_loss_tracker.update_state(recon)

        y_true_2d = tf.reshape(tf.cast(y, tf.float32), [tf.shape(y)[0], -1])
        y_pred_2d = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])

        self.mae.update_state(y_true_2d, y_pred_2d, sample_weight=w)
        self.mse.update_state(y_true_2d, y_pred_2d, sample_weight=w)

        return {m.name: m.result() for m in self.metrics}


# ──────────────────────────────────────────────────────────────────────────────
# tf.data helper
# ──────────────────────────────────────────────────────────────────────────────

def make_ds(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    repeat: bool,
    drop_remainder: bool,
) -> tf.data.Dataset:
    x_dict = {
        "x": np.asarray(X, dtype=np.float32),
        "d": np.asarray(d, dtype=np.int32),
        "w": np.asarray(w, dtype=np.float32),
    }
    y = np.asarray(y, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x_dict, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 50000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────────────────────────────────────
# training entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_training(
    base_model: tf.keras.Model,
    cfg: EngineConfig,
    X_tr: np.ndarray, y_tr: np.ndarray, d_tr: np.ndarray, w_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, d_val: np.ndarray, w_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, d_test: np.ndarray, w_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    val_steps: Optional[int] = None,
    lr: float = 1e-3,
    save_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    mixed_precision_enable: bool = False,
    extra_eval: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:

    if mixed_precision_enable:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    engine = CoralTrainingModel(base=base_model, cfg=cfg)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if mixed_precision_enable:
        try:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        except Exception:
            pass

    engine.compile(optimizer=opt, run_eagerly=False)

    ds_tr = make_ds(X_tr, y_tr, d_tr, w_tr, batch_size, shuffle=True,  seed=seed, repeat=True,  drop_remainder=True)
    # keep val deterministic but shuffled once in Main.py (we don’t shuffle here)
    ds_va = make_ds(X_val, y_val, d_val, w_val, batch_size, shuffle=False, seed=seed, repeat=True,  drop_remainder=True)
    ds_te = make_ds(X_test, y_test, d_test, w_test, batch_size, shuffle=False, seed=seed, repeat=False, drop_remainder=False)

    if val_steps is None:
        val_steps = max(1, len(X_val) // batch_size)

    hist = engine.fit(
        ds_tr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_va,
        validation_steps=val_steps,
        callbacks=(callbacks or []),
        verbose=2,
    )

    test_metrics = engine.evaluate(ds_te, verbose=0, return_dict=True)

    extra_results: Dict[str, Dict[str, float]] = {}
    if extra_eval:
        for name, (X_e, y_e, d_e, w_e) in extra_eval.items():
            ds_e = make_ds(X_e, y_e, d_e, w_e, batch_size, shuffle=False, seed=seed, repeat=False, drop_remainder=False)
            extra_results[name] = {k: float(v) for k, v in engine.evaluate(ds_e, verbose=0, return_dict=True).items()}

    cfg_dict = asdict(cfg)
    cfg_dict["coral_lambda"] = _lambda_float(cfg.coral_lambda)  # NEW: make JSON-safe

    out: Dict[str, Any] = {
        "config": cfg_dict,
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
        "extra": extra_results,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            base_model.save_weights(str(save_dir / "base_model.weights.h5"))
        except Exception:
            pass
        with open(save_dir / "training_report.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    return out
