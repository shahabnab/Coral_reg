# Main.py
# ============================================================
# Multi-source -> Target Domain Adaptation (UDA) using Deep CORAL
# REGRESSION task: predict "error ratio"
#
# Per trial it prints:
#   - ADAPTION: MAE/MSE
#   - test_adaption: MAE/MSE
#   - TEST: MAE/MSE
# And saves:
#   - eval_report.xlsx
#   - eval_summary.csv + eval_*.csv
#   - plots_*/*.png
# ============================================================

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Global label mode (set once, used everywhere) ---
# "ratio": y = (sensor-camera)/sensor
# "error": y = (sensor-camera)  (meters)
CORAL_LABEL_MODE = "ratio"  # <-- change to "error" when needed
os.environ["CORAL_LABEL_MODE"] = CORAL_LABEL_MODE
# ----------------------------------------------------

import tensorflow as tf

import optuna
import optuna.visualization.matplotlib as ovm
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

from my_helping_functions import set_seed
from my_df_processing import download_dataset, slicing_dts
from my_cir_processing import CIR_pipeline
from utilities import *
from NNmodel import *



try:
    from my_helping_functions import combine_trial_reports
except Exception:
    combine_trial_reports = None

from training_engine import EngineConfig, run_training

# NEW: reporting helper file
from coral_reporting import save_eval_excel_and_plots,save_range_metrics_excel







def save_parallel_wide(study, out_dir, params=None,
                       inch_per_param=2.0, height=7, max_width_in=40, dpi=240):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    if params is None:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        keys = sorted({k for t in completed for k in t.params})
        varying = []
        for k in keys:
            vals = [t.params[k] for t in completed if k in t.params]
            if not vals:
                continue
            try:
                if max(vals) != min(vals):
                    varying.append(k)
            except TypeError:
                if len(set(vals)) > 1:
                    varying.append(k)
        params = varying or keys

    ax = ovm.plot_parallel_coordinate(study, params=params)
    fig = ax.figure
    width = min(max_width_in, max(12, inch_per_param * len(params)))
    fig.set_size_inches(width, height)

    for a in fig.axes:
        a.tick_params(axis="x", labelrotation=35, labelsize=10)
        a.tick_params(axis="y", labelsize=9)
        for line in getattr(a, "lines", []):
            line.set_alpha(0.35)
            line.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(out / "parallel_coordinate_wide.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)




def run_optuna(objective_fn, s_d, SEED, SAVE_PLOTS_ROOT, CONFIG_ATTR, WHOLE_RES_XLSX,
               n_trials=60, direction="minimize"):

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=2)
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=20),
        storage=f"sqlite:///{SAVE_PLOTS_ROOT / 'optuna_study.db'}",
        pruner=pruner,
    )

    study.optimize(objective_fn, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True, catch=(Exception,))

    print("\n═════════ Best trial ═════════")
    print("Trial # :", study.best_trial.number)
    print("Score   :", study.best_value)
    print("Params  :", study.best_params)

    records = {
        f"trial_{t.number:03d}": t.user_attrs[CONFIG_ATTR]
        for t in study.trials
        if t.state == TrialState.COMPLETE and CONFIG_ATTR in t.user_attrs
    }
    if records:
        (pd.DataFrame(records).T.reset_index().rename(columns={"index": "trial_name"})
         .to_excel(SAVE_PLOTS_ROOT / WHOLE_RES_XLSX, index=False))
        print(f"\n📝 Saved {len(records)} configs → {WHOLE_RES_XLSX}")

    study.trials_dataframe().to_csv(SAVE_PLOTS_ROOT / "optuna_results.csv", index=False)
    df = study.trials_dataframe(attrs=("number","state","value","params","datetime_start","datetime_complete","duration","user_attrs"))
    df.to_excel(SAVE_PLOTS_ROOT / "trials.xlsx", index=False)

    try:
        save_parallel_wide(study, SAVE_PLOTS_ROOT)
    except Exception as e:
        print(f"⚠️ Could not plot parallel coords: {e}")

    if combine_trial_reports is not None:
        try:
            combine_trial_reports(SAVE_PLOTS_ROOT)
        except Exception:
            pass


# ───────────────────────────── scenario runner ─────────────────────────────

def run_scenario(s_d):
    soft_reset(tag="start scenario")

    print(f'\n=== Scenario: {s_d["scenario_path"]} | SOURCES={s_d["Training_datasets"]} -> TARGET={s_d["Test_dataset"]} ===')

    tf.random.set_seed(s_d["SEED"])
    set_seed(s_d["SEED"])

    SAVE_PLOTS_ROOT = Path(s_d["scenario_path"])
    SAVE_PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    WHOLE_RES_XLSX = "whole_res.xlsx"
    CONFIG_ATTR = "config"

    # download + slice
    raw_datasets = download_dataset(s_d["Training_datasets"], s_d["Test_dataset"], s_d["SEED"])
    balanced_dtsets = slicing_dts(raw_datasets, s_d)
    #extracting the required data

    # extract arrays
    CIRS      = CIR_pipeline(balanced_dtsets)

    # --- use the global mode to select the right label column + eval key ---
    reg_col = "error ratio" if CORAL_LABEL_MODE == "ratio" else "error"
    y_pack_key = "y_ratio" if CORAL_LABEL_MODE == "ratio" else "y_error"

    RegLabels = take_reg_labels(balanced_dtsets, col=reg_col)
    # -------------------------------------------------------------

    Domains   = take_domains_local(balanced_dtsets, s_d)
    Weights   = take_weights_local(balanced_dtsets, s_d)

    # helper to pack eval split including ranges
    def pack_split(split_key: str):
        df = balanced_dtsets[split_key]
        X = np.asarray(CIRS[split_key])[..., None].astype("float32", copy=True)
        y = np.asarray(RegLabels[split_key]).astype(np.float32, copy=True)
        d = np.asarray(Domains[split_key]).astype(np.int32, copy=True)
        w = np.ones_like(y, dtype=np.float32)

        sensor_rng = df["Sensor rng"].to_numpy(dtype=np.float32, copy=True)
        camera_rng = df["camera rng"].to_numpy(dtype=np.float32, copy=True)
        return X, y, d, w, sensor_rng, camera_rng

    # train pool = TRAIN* + ADAPTION (exclude test_adaption + TEST)
    train_roles = s_d["DATASET_ROLES"][0:-2]

    X = np.concatenate([CIRS[k] for k in train_roles])[..., None].astype("float32", copy=True)
    d = np.concatenate([Domains[k] for k in train_roles]).astype(np.int32, copy=True)
    w = np.concatenate([Weights[k] for k in train_roles]).astype(np.float32, copy=True)
    y = np.concatenate([RegLabels[k] for k in train_roles]).astype(np.float32, copy=True)

    # pack eval splits for each split "Adaption", "test_adaption", "TEST"
    X_ad, y_ad, d_ad, w_ad, s_ad, c_ad = pack_split("ADAPTION")
    X_ta, y_ta, d_ta, w_ta, s_ta, c_ta = pack_split("test_adaption")
    X_te, y_te, d_te, w_te, s_te, c_te = pack_split("TEST")

    # split train/val for training process
    X_tr, X_val, d_tr, d_val, w_tr, w_val, y_tr, y_val = train_test_split(
        X, d, w, y,
        test_size=0.30,
        random_state=s_d["SEED"],
        stratify=d,
        shuffle=True
    )

    # shuffle val once so batches are mixed
    rs = np.random.RandomState(s_d["SEED"])
    p = rs.permutation(len(X_val))
    X_val, d_val, w_val, y_val = X_val[p], d_val[p], w_val[p], y_val[p]

    input_shape = X_tr.shape[1:]  # (T,1)

    # CORAL ids
    #specifying the source and target domain ids for coral loss
    source_ids = tuple(range(len(s_d["Training_datasets"])))
    target_id  = int(s_d["PL_DOMAIN_ID"])

    cfg_base = EngineConfig(
        task_type="reg",
        adaption_with_label=s_d["ADAPTION_WITH_LABEL"],
        use_coral=True,
        coral_lambda=0.2,
        coral_target_id=target_id,
        coral_source_ids=source_ids,
        grad_clipnorm=5.0,
        output_pred_key="pred",
        output_latent_key="latent",
        output_recon_key="recon",   # NEW
        recon_lambda=0.1,           # NEW default (overridden per-trial below)
        use_recon=True,             # NEW
    )

    class EvalSplitTrends(tf.keras.callbacks.Callback):
        """
        Evaluate a fixed split (e.g., test_adaption) at each epoch end using inference only.
        Computes MAE/MSE without calling model.evaluate() (avoids touching Keras metric state).
        """
        def __init__(self, *, name: str, X, y, d, w, batch_size: int, pred_key: str = "pred", verbose: int = 0):
            super().__init__()
            self.name_ = str(name)
            self.X = np.asarray(X, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.float32)
            self.d = np.asarray(d, dtype=np.int32)
            self.w = np.asarray(w, dtype=np.float32)
            self.batch_size = int(batch_size)
            self.pred_key = str(pred_key)
            self.verbose = int(verbose)
            self.ds = None

        def on_train_begin(self, logs=None):
            x_dict = {"x": self.X, "d": self.d, "w": self.w}
            self.ds = tf.data.Dataset.from_tensor_slices((x_dict, self.y)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        def _get_base(self):
            # training_engine wraps the base model as `self.model.base`; fallback to `self.model`.
            return getattr(self.model, "base", None) or self.model

        def _extract_pred(self, outputs):
            if isinstance(outputs, dict):
                return outputs.get(self.pred_key, next(iter(outputs.values())))
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                return outputs[0]
            return outputs

        def on_epoch_end(self, epoch, logs=None):
            base = self._get_base()

            sum_abs = tf.constant(0.0, tf.float32)
            sum_sq  = tf.constant(0.0, tf.float32)
            sum_w   = tf.constant(0.0, tf.float32)

            for x_dict, y_true in self.ds:
                y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
                w = tf.reshape(tf.cast(x_dict.get("w", tf.ones_like(y_true)), tf.float32), [-1])

                outputs = base(x_dict["x"], training=False)
                y_pred = self._extract_pred(outputs)
                y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

                err = y_true - y_pred
                sum_abs += tf.reduce_sum(tf.abs(err) * w)
                sum_sq  += tf.reduce_sum(tf.square(err) * w)
                sum_w   += tf.reduce_sum(w)

            mae = tf.math.divide_no_nan(sum_abs, sum_w)
            mse = tf.math.divide_no_nan(sum_sq,  sum_w)

            mae_f = float(mae.numpy())
            mse_f = float(mse.numpy())

            if logs is not None:
                logs[f"{self.name_}_mae"] = mae_f
                logs[f"{self.name_}_mse"] = mse_f

            if self.verbose:
                print(f"[TREND] {self.name_:>12} | mae={mae_f:.6f}  mse={mse_f:.6f}")

    def objective_coral(trial, force_epochs=None, force_batch=None, save_tag=""):
        epochs = int(force_epochs or trial.suggest_int("EPOCHS", 20, 120))
        batch  = int(force_batch  or trial.suggest_categorical("BATCH", [64, 128, 256]))
        lr     = float(trial.suggest_float("LR", 1e-5, 3e-3, log=True))

        coral_lambda = float(trial.suggest_float("CORAL_LAMBDA", 0.0, 2.0))
        recon_lambda = float(trial.suggest_float("RECON_LAMBDA", 0.0, 1.0))  # NEW

        latent_dim   = int(trial.suggest_categorical("LATENT_DIM", [32, 64, 128, 256]))
        base_filters = int(trial.suggest_categorical("BASE_FILTERS", [16, 32, 48, 64]))
        dropout      = float(trial.suggest_float("DROPOUT", 0.0, 0.35))

        # NEW: ramp-up (0 disables)
        coral_ramp_epochs = int(trial.suggest_categorical("CORAL_RAMP_EPOCHS", [0, 5, 10, 20, 30]))

        cfg = EngineConfig(**cfg_base.__dict__)
        cfg.recon_lambda = recon_lambda  # NEW

        # NEW: use a tf.Variable so CoralLambdaRampUp can update lambda during fit()
        lambda_var = tf.Variable(
            0.0 if (coral_ramp_epochs > 0 and coral_lambda > 0.0) else coral_lambda,
            dtype=tf.float32,
            trainable=False,
            name="coral_lambda",
        )
        cfg.coral_lambda = lambda_var

        base_model = build_coral_cir_reg_model(  # CHANGED: AE model
            input_shape=input_shape,
            latent_dim=latent_dim,
            base_filters=base_filters,
            dropout=dropout,
        )

        steps_per_epoch = max(1, len(X_tr) // batch)
        val_steps       = max(1, len(X_val) // batch)

        out_dir = SAVE_PLOTS_ROOT / (save_tag or f"trial_{trial.number:03d}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # NEW: ramp callback (only if enabled)
        callbacks = []
        if coral_ramp_epochs > 0 and coral_lambda > 0.0:
            callbacks.append(
                CoralLambdaRampUp(
                    lambda_var=lambda_var,
                    max_lambda=coral_lambda,
                    rampup_epochs=coral_ramp_epochs,
                    mode="sigmoid",
                    verbose=0,
                )
            )

        # NEW: epoch-wise trends on test_adaption (no training influence)
        callbacks.append(
            EvalSplitTrends(
                name="test_adaption",
                X=X_ta, y=y_ta, d=d_ta, w=w_ta,
                batch_size=batch,
                pred_key=cfg.output_pred_key,
                verbose=0,
            )
        )

        report = run_training(
            base_model=base_model,
            cfg=cfg,
            X_tr=X_tr, y_tr=y_tr, d_tr=d_tr, w_tr=w_tr,
            X_val=X_val, y_val=y_val, d_val=d_val, w_val=w_val,
            X_test=X_te, y_test=y_te, d_test=d_te, w_test=w_te,
            epochs=epochs,
            batch_size=batch,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            lr=lr,
            save_dir=out_dir,
            seed=s_d["SEED"],
            callbacks=callbacks,  # <-- was []
            mixed_precision_enable=False,
            extra_eval={
                "ADAPTION": (X_ad, y_ad, d_ad, w_ad),
                "test_adaption": (X_ta, y_ta, d_ta, w_ta),
            }
        )

        # print evaluations each run
        ex = report.get("extra", {})
        te = report.get("test", {})

        if "ADAPTION" in ex:
            print(f"[EVAL] ADAPTION      | mae={ex['ADAPTION'].get('mae', np.nan):.6f}  mse={ex['ADAPTION'].get('mse', np.nan):.6f}")
        if "test_adaption" in ex:
            print(f"[EVAL] test_adaption | mae={ex['test_adaption'].get('mae', np.nan):.6f}  mse={ex['test_adaption'].get('mse', np.nan):.6f}")
        print(f"[EVAL] TEST          | mae={te.get('mae', np.nan):.6f}  mse={te.get('mse', np.nan):.6f}")

        # ---- Save excel + plots (DO NOT fail trial if reporting fails) ----
        splits_pack = {
            "ADAPTION": {
                "X": X_ad, y_pack_key: y_ad,
                "sensor_rng": s_ad, "camera_rng": c_ad,
                "domain_id": d_ad,
            },
            "test_adaption": {
                "X": X_ta, y_pack_key: y_ta,
                "sensor_rng": s_ta, "camera_rng": c_ta,
                "domain_id": d_ta,
            },
            "TEST": {
                "X": X_te, y_pack_key: y_te,
                "sensor_rng": s_te, "camera_rng": c_te,
                "domain_id": d_te,
            },
        }

        try:
            model_obj = report.get("model", None)
            model_path = report.get("model_path", None)

            if model_obj is None and model_path is None:
                model_obj = base_model

            save_eval_excel_and_plots(
                save_dir=out_dir,
                model_or_path=(model_obj if model_obj is not None else model_path),
                splits=splits_pack,
                batch_size=batch,
                excel_name="eval_report.xlsx",
                label_mode=CORAL_LABEL_MODE,  # make reporting deterministic
                y_key=y_pack_key,             # avoid relying on default key names
            )

            save_range_metrics_excel(
                save_dir=out_dir,
                model_or_path=(model_obj if model_obj is not None else model_path),
                splits=splits_pack,
                batch_size=batch,
                excel_name="range_metrics.xlsx",
                label_mode=CORAL_LABEL_MODE,  # make range correction math match mode
            )
        except Exception as e:
            print(f"⚠️ Reporting failed (trial continues): {e}")

        # Optuna score: minimize val_mae (safe)
        hist = report.get("history", {})
        score = float(hist["val_mae"][-1]) if "val_mae" in hist and len(hist["val_mae"]) else float(te.get("mae", 1e9))

        trial.set_user_attr(CONFIG_ATTR, {
            "scenario": s_d["scenario_path"],
            "params": dict(trial.params),
            "test": te,
            "extra": ex,
            "score_used": score,
            "coral_lambda_target": coral_lambda,
            "coral_ramp_epochs": coral_ramp_epochs,
            "recon_lambda": recon_lambda,  # NEW
        })

        return score

    run_optuna(
        objective_fn=objective_coral,
        s_d=s_d,
        SEED=s_d["SEED"],
        SAVE_PLOTS_ROOT=SAVE_PLOTS_ROOT,
        CONFIG_ATTR=CONFIG_ATTR,
        WHOLE_RES_XLSX=WHOLE_RES_XLSX,
        n_trials=s_d["Trials"],
        direction="minimize",
    )


# ───────────────────────────── main ─────────────────────────────

if __name__ == "__main__":
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print(f"⚠️ Could not set memory growth: {e}")
    #setting the seed to have reproducible results
    # set the number of trials    

    Seed = 42
    Trials = 100
    print("Using fixed seed:", Seed)
    # dataset sizes for training

    DATASET_CONFIG = {
        "TU":     6500,
        "Office": 8000,
        "IOT":    7000,
    }

    # Scenarios: [source1, source2, ..., target]
    scens = [
        ["TU", "IOT", "Office"],
        ["IOT", "Office", "TU"],
        ["TU", "Office", "IOT"],
    ]

    for scen in scens:
        soft_reset(tag=f"scenario {scen}")

        ADAPTION_WITH_LABEL = 0
        Training_datasets = scen[:-1]
        Test_dataset = scen[-1]
        PL_DOMAIN_ID = len(Training_datasets)

        DATASET_ROLES = [f"TRAIN{i+1}" for i in range(len(Training_datasets))]
        DATASET_ROLES += ["ADAPTION", "test_adaption", "TEST"]

        scenario_path = "_".join(Training_datasets) + f"_to_{Test_dataset}"
        Training_labels = [f"TRAIN{i+1}" for i in range(len(Training_datasets))]

        s_d = {
            "SEED": Seed,
            "Trials": Trials,

            "Training_labels": Training_labels,
            "Training_datasets": Training_datasets,
            "Test_dataset": Test_dataset,
            "train_size": DATASET_CONFIG.get(Test_dataset, 6500),

            "scenario_path": scenario_path,
            "DATASET_ROLES": DATASET_ROLES,

            "ADAPTION_WITH_LABEL": ADAPTION_WITH_LABEL,
            "PL_DOMAIN_ID": PL_DOMAIN_ID,
        }

        print("\n------------------------------")
        print("Scenario path:", scenario_path)
        print("Sources:", Training_datasets, "| Target:", Test_dataset)
        print("PL_DOMAIN_ID:", PL_DOMAIN_ID)
        print("------------------------------\n")

        run_scenario(s_d)
