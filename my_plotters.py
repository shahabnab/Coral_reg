# ===== Standard library =====
import math
import os

# ===== Third-party =====
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

import pickle
def prob_class1(y):  # softmax (N,2) or sigmoid (N,1)/(N,)
    y = np.asarray(y)
    return y[:, 1] if (y.ndim == 2 and y.shape[-1] == 2) else y.squeeze()

def argmax_labels(y_soft):  # → integer labels 0/1
    y = np.asarray(y_soft)
    if y.ndim == 2 and y.shape[-1] == 2:     # softmax
        return np.argmax(y, axis=1).astype(int)
    else:                                     # sigmoid
        return (y.reshape(-1) >= 0.5).astype(int)


def plot_latent_umap(encoder_model,dt_roles, CIRS, Domains,save_plots, title="UMAP of Latent Space"):
    list_roles=dt_roles[0:-1]
    X = np.concatenate([CIRS[k] for k in list_roles])[..., None]
    d = np.concatenate([Domains[k] for k in list_roles])

    latent_vectors = encoder_model.predict(X)
    scaled_latents = StandardScaler().fit_transform(latent_vectors)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_latents)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=d, palette="Set2", s=30, alpha=0.7)
    #plt.title(title)
    plt.xlabel("UMAP-1", fontsize=13,fontweight="bold")
    plt.ylabel("UMAP-2", fontsize=13,fontweight="bold")
    plt.legend(title="Domain", fontsize=10, markerscale=1.8,handlelength=1.6)
    plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.savefig(os.path.join(save_plots, "latent_umap_latent_vector.png"))
    plt.close()




def plot_latent_umap_input( CIRS,dt_roles, Domains,save_plots, title="UMAP of Latent Space"):
    list_roles=dt_roles[0:-1]
    X = np.concatenate([CIRS[k] for k in list_roles])
    d = np.concatenate([Domains[k] for k in list_roles])

    # Flatten each sample if X is 2D (samples, timesteps), else reshape appropriately
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X

    scaled_input = StandardScaler().fit_transform(X_flat)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_input)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=d, palette="Set2", s=30, alpha=0.7)
    #plt.title(title)
    plt.xlabel("UMAP-1", fontsize=13,fontweight="bold")
    plt.ylabel("UMAP-2", fontsize=13,fontweight="bold")
    plt.legend(title="Domain", fontsize=10, markerscale=1.8,handlelength=1.6)
    plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.savefig(os.path.join(save_plots, "latent_umap_input.png"))
    plt.close()  






def plot_latent_umap(encoder_model, dt_roles,CIRS, Domains, save_plots, title="UMAP of Latent Space"):
    list_roles=dt_roles[0:-1]
    X = np.concatenate([CIRS[k] for k in list_roles])[..., None]
    d = np.concatenate([Domains[k] for k in list_roles])

    latent_vectors = encoder_model.predict(X, verbose=0)
    scaled_latents = StandardScaler().fit_transform(latent_vectors)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_latents)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=d, palette="Set2", s=30, alpha=0.7)
    #plt.title(title)
    plt.xlabel("UMAP-1", fontsize=13,fontweight="bold")
    plt.ylabel("UMAP-2",fontsize=13,fontweight="bold")
    plt.legend(title="Domain", fontsize=10, markerscale=1.8,handlelength=1.6)
    plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
    plt.gca().set_axisbelow(True)
    os.makedirs(save_plots, exist_ok=True)
    plt.savefig(os.path.join(save_plots, "latent_umap_latent_vector.png"), bbox_inches="tight", dpi=160)
    plt.close()



# ---------- LATENT (encoder output) ----------
def plot_latent_umap_by_los(encoder_model,dt_roles, CIRS, Domains, LOS, save_plots,
                                     title="UMAP of Latent Space",
                                     n_neighbors=20, min_dist=0.1, metric="euclidean"):
    os.makedirs(save_plots, exist_ok=True)
    list_roles=dt_roles[0:-1]

    X = np.concatenate([CIRS[k] for k in list_roles])[..., None]
    d = np.concatenate([Domains[k] for k in list_roles])

    if isinstance(LOS, dict):
        y = np.concatenate([LOS[k] for k in list_roles]).ravel()
    else:
        y = np.asarray(LOS).ravel()

    # Encode -> scale -> UMAP (fit ONCE on all points)
    Z = encoder_model.predict(X, verbose=0)
    Z = StandardScaler().fit_transform(Z)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    E = reducer.fit_transform(Z)  # (N, 2)

    # Build DF
    domain_series = pd.Series(d).astype(str)
    domain_order = sorted(domain_series.unique(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)
    df = pd.DataFrame({
        "umap1": E[:, 0],
        "umap2": E[:, 1],
        "domain": domain_series.values,
        "los": np.where(y >= 0.5, "LOS", "NLOS")
    })

    # Global limits for identical axes
    xpad = 0.05*(df["umap1"].max() - df["umap1"].min())
    ypad = 0.05*(df["umap2"].max() - df["umap2"].min())
    xlim = (df["umap1"].min() - xpad, df["umap1"].max() + xpad)
    ylim = (df["umap2"].min() - ypad, df["umap2"].max() + ypad)

    # Save two separate figures
    for cls in ("LOS", "NLOS"):
        sub = df[df["los"] == cls]
        plt.figure(figsize=(9,7))
        sns.scatterplot(
            data=sub, x="umap1", y="umap2",
            hue="domain", hue_order=domain_order, palette="Set2",
            s=32, alpha=0.7
        )
        plt.title(f"")
        plt.xlabel("UMAP-1", fontsize=13, fontweight="bold"); plt.ylabel("UMAP-2", fontsize=13, fontweight="bold")
        plt.xlim(*xlim); plt.ylim(*ylim)
        plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        plt.gca().set_axisbelow(True)
        plt.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, markerscale=1.8,handlelength=1.6)
        plt.tight_layout()
        base = os.path.join(save_plots, f"latent_umap_{cls.lower()}_only")
        plt.savefig(base + ".png", dpi=200, bbox_inches="tight")
       
        plt.close()


# ---------- INPUT (raw signals) ----------
def plot_umap_input_by_los(CIRS,dt_roles, Domains, LOS, save_plots,
                                    title="UMAP of Inputs",
                                    n_neighbors=20, min_dist=0.1, metric="euclidean"):
    os.makedirs(save_plots, exist_ok=True)
    list_roles=dt_roles[0:-1]

    X = np.concatenate([CIRS[k] for k in list_roles])
    d = np.concatenate([Domains[k] for k in list_roles])

    if isinstance(LOS, dict):
        y = np.concatenate([LOS[k] for k in list_roles]).ravel()
    else:
        y = np.asarray(LOS).ravel()

    # Flatten to 2D
    # Flatten to 2D
    X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else (X if X.ndim == 2 else X[:, None])

    # Scale -> UMAP (fit ONCE on all points)
    Z = StandardScaler().fit_transform(X_flat)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    E = reducer.fit_transform(Z)

    # Build DF
    domain_series = pd.Series(d).astype(str)
    domain_order = sorted(domain_series.unique(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)
    df = pd.DataFrame({
        "umap1": E[:, 0],
        "umap2": E[:, 1],
        "domain": domain_series.values,
        "los": np.where(y >= 0.5, "LOS", "NLOS")
    })

    # Global limits identical for both
    xpad = 0.05*(df["umap1"].max() - df["umap1"].min())
    ypad = 0.05*(df["umap2"].max() - df["umap2"].min())
    xlim = (df["umap1"].min() - xpad, df["umap1"].max() + xpad)
    ylim = (df["umap2"].min() - ypad, df["umap2"].max() + ypad)

    # Save two separate figures
    for cls in ("LOS", "NLOS"):
        sub = df[df["los"] == cls]
        plt.figure(figsize=(9,7))
        sns.scatterplot(
            data=sub, x="umap1", y="umap2",
            hue="domain", hue_order=domain_order, palette="Set2",
            s=32, alpha=0.7
        )
        plt.title("")
        plt.xlabel("UMAP-1",fontsize=13,fontweight="bold"); plt.ylabel("UMAP-2",fontsize=13,fontweight="bold")
        plt.xlim(*xlim); plt.ylim(*ylim)
        plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        plt.gca().set_axisbelow(True)
        plt.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        base = os.path.join(save_plots, f"umap_input_{cls.lower()}_only")
        plt.savefig(base + ".png", dpi=200, bbox_inches="tight")

        plt.close()



def signal_entropy(signal):
    """Compute normalized Shannon entropy of a signal."""
    prob_dist, _ = np.histogram(signal, bins=32, density=True)
    prob_dist = prob_dist[prob_dist > 0]  # avoid log(0)
    return entropy(prob_dist) / np.log(len(prob_dist))

def compute_complexity_metrics(CIRS):
    metrics = {
        "domain": [],
        "variance": [],
        "peak_to_mean": [],
        "entropy": [],
    }

    for domain_name, signals in CIRS.items():
        for sig in signals:
            metrics["domain"].append(domain_name)
            metrics["variance"].append(np.var(sig))
            metrics["peak_to_mean"].append(np.max(sig) / (np.mean(sig) + 1e-6))
            metrics["entropy"].append(signal_entropy(sig))

    return metrics

def plot_complexity_metrics(CIRS):
    """Compute and plot complexity metrics for each domain in CIRS."""
    metrics = compute_complexity_metrics(CIRS)
    df_metrics = pd.DataFrame(metrics)
    # Boxplots for comparison
    plt.figure(figsize=(14, 4))
    for i, metric in enumerate(["variance", "peak_to_mean", "entropy"]):
        plt.subplot(1, 3, i+1)
        sns.boxplot(data=df_metrics, x="domain", y=metric)
        plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        plt.title(metric.capitalize())
    plt.tight_layout()
    plt.savefig("complexity_metrics.png")
    plt.close() 




def plot_encoded_signals_pro(ae, decoder, encoder, CIRS, labels, save_plots,
                             splits=("TRAIN1","TRAIN2","ADAPTION","TEST"),
                             los_map={0: "LOS", 1: "NLOS"},
                             dpi=220):
    """
    CIRS:   dict of split -> np.ndarray, shape (N, T) or (N, T, 1)
    labels: dict of split -> np.ndarray of {0,1} where 0=LOS, 1=NLOS (adjust los_map if different)

    Creates two multi-panel figures: one for LOS, one for NLOS (2x2: TRAIN1, TRAIN2, ADAPTION, TEST).
    Each panel overlays Input vs Reconstruction and annotates MAE / RMSE.
    """
    os.makedirs(save_plots, exist_ok=True)

    # -------- helpers
    def _ensure_1d(sig):
        sig = np.asarray(sig)
        if sig.ndim == 1:
            return sig
        if sig.ndim == 2 and sig.shape[-1] == 1:
            return sig[:, 0]
        if sig.ndim == 2 and sig.shape[0] == 1:
            return sig[0]
        if sig.ndim == 3 and sig.shape[-1] == 1:
            return sig[0, :, 0] if sig.shape[0] == 1 else sig[0, :, 0]
        return sig.squeeze()

    def reconstruct_one(x1d):
        """encoder -> decoder; robust to (T) or (T,1)"""
        x = np.asarray(x1d)
        if x.ndim == 1:
            xin = x[None, :, None]        # (1, T, 1)
        elif x.ndim == 2:
            xin = x[None, ...]            # (1, T, 1) or (1, T)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Prefer encoder->decoder if provided; otherwise fall back to ae
        if encoder is not None and decoder is not None:
            z = encoder.predict(xin, verbose=0)
            y = decoder.predict(z, verbose=0)
        else:
            y = ae.predict(xin, verbose=0)
        return _ensure_1d(y[0])

    # -------- pick one example per split/class and precompute recon
    selections = {}   # key=(split, "LOS"/"NLOS") -> (raw, rec)
    all_vals = []
    for split in splits:
        X = CIRS.get(split, None)
        y = labels.get(split, None)
        if X is None or y is None:
            continue

        # Ensure (N, T)
        X = np.asarray(X)
        if X.ndim == 3 and X.shape[-1] == 1:
            X = X[:, :, 0]

        for cls_val, cls_name in los_map.items():
            idx = np.where(y == cls_val)[0]
            if idx.size == 0:
                continue
            raw = _ensure_1d(X[idx[0]])
            rec = reconstruct_one(raw)
            selections[(split, cls_name)] = (raw, rec)
            all_vals.extend([raw.min(), raw.max(), rec.min(), rec.max()])

    if not selections:
        raise ValueError("No samples found to plot. Check your splits and labels mapping.")

    # global y-limits for comparability
    ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ylim = (ymin - pad, ymax + pad)

    # -------- common small style tweaks
    def _beautify_ax(ax):
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=13)

    # -------- builder for a 2x2 figure (LOS or NLOS)
    def make_figure(target_cls,splits):
        n = len(splits)

        if n==5:
            fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
        elif n==4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        elif n==3:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)

        axes = axes.ravel()
     

        # Plot each split in a cell (or hide if missing)
        handles_cache = None
        for i, split in enumerate(splits):
            ax = axes[i]
            pair = selections.get((split, target_cls))
            if pair is None:
                ax.axis("off")
                continue

            raw, rec = pair
            err = raw - rec
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))

            # Plot
            ax.plot(raw, linewidth=1.6, label="Input")
            ax.plot(rec, linewidth=1.4, linestyle="--", label="Reconstruction")
            ax.set_title(f"{split} — {target_cls}\nMAE={mae:.3g} | RMSE={rmse:.3g}", fontsize=13,fontweight="bold")
            ax.set_ylim(*ylim)
            _beautify_ax(ax)

            if handles_cache is None:
                handles_cache = ax.get_legend_handles_labels()

        # shared labels & legend
        fig.supxlabel("Sample Index", fontsize=13,fontweight="bold")
        fig.supylabel("Amplitude", fontsize=13,fontweight="bold")
        if handles_cache is not None:
            fig.legend(*handles_cache, loc="upper right", ncol=2, frameon=False)

        fig.suptitle("Input vs. Reconstruction", y=0.995, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # save
        base = f"reconstruction_overlay_{target_cls.lower()}"
        fig.savefig(os.path.join(save_plots, f"{base}.png"), dpi=dpi, bbox_inches="tight")
        
        plt.close(fig)

    # Build LOS and NLOS figures (only if present)
    for cls in set(name for _, name in los_map.items()):
        # make figure only if we have at least one panel for this class
        if any(k[1] == cls for k in selections.keys()):
            try:
                make_figure(cls,splits)
            except Exception as e:
                print(f"Error making figure for class {cls}: {e}")  


def probe_ae(ae, CIRS, labels, h):
    
    save_plots=h.get("save_plots")
    os.makedirs(save_plots, exist_ok=True)
    CIRS_test_LOS= CIRS["TEST"][labels["TEST"]==0][..., None]
    CIRS_test_NLOS= CIRS["TEST"][labels["TEST"]==1][..., None]
    for i in range(1, 2):
        print("CIRS_test_LOS shape:", CIRS_test_LOS[i].shape)
        print("CIRS_test_NLOS shape:", CIRS_test_NLOS[i].shape)
        # 1) Prepare LOS / NLOS with a proper batch dimension of 1
        x_los  = CIRS_test_LOS [i:i+1]   # shape: (1, 150, 1)
        x_nlos = CIRS_test_NLOS[i:i+1]   # shape: (1, 150, 1)

        # 2) Layers to probe
        """ probe_layer_names = [
            "latent_vector",      # 16-dim bottleneck
            "dense",              # 64-unit Dense (replace latent_cls)
            "multiply",           # Gated output before los_logits (replace latent_attention)
            "los_logits",         # final 1-dim sigmoid
        ] """
        probe_layer_names = [
            "latent_vector",
            "latent_cls",
            "attention_gated",
            "los_prob"
        ]

        # 3) Build a probe model
        probe_outputs = [ae.get_layer(name).output
                        for name in probe_layer_names]
        probe_model   = Model(inputs=ae.input,
                            outputs=probe_outputs)

        # 4) Run the probe
        los_vec, los_cls, los_attn, los_logits  = probe_model.predict(x_los)
        nlos_vec, nlos_cls, nlos_attn, nlos_logits = probe_model.predict(x_nlos)

        # 5) Squeeze away the batch axis
        los_vec    = los_vec.squeeze()     # → (16,)
        los_cls    = los_cls.squeeze()     # → (64,)
        los_attn   = los_attn.squeeze()    # → (64,)
      

        #los_logits = float(los_logits)     # → scalar
        los_logits= float(prob_class1(los_logits))

        nlos_vec    = nlos_vec.squeeze()
        nlos_cls    = nlos_cls.squeeze()
        nlos_attn   = nlos_attn.squeeze()
      
        #nlos_logits = float(nlos_logits)
        nlos_logits=float(prob_class1(nlos_logits))

        # 6) Plot everything
        fig, axes = plt.subplots(4, 1, figsize=(6, 12))

        # ── latent_vector (16 dims)
        axes[0].plot(los_vec,   marker='o', color='blue',   label='LOS')
        axes[0].plot(nlos_vec,  marker='x', linestyle='--', color='orange', label='NLOS')
        axes[0].set_title("latent_vector (16 dims)")
        axes[0].set_ylabel("Activation")
        axes[0].legend()

        # ── latent_cls (64 dims)
        axes[1].plot(los_cls,   marker='o', color='blue')
        axes[1].plot(nlos_cls,  marker='x', linestyle='--', color='orange')
        axes[1].set_title("latent_cls (64 dims)")
        axes[1].set_ylabel("Activation")

        # ── latent_attention (64 dims)
        axes[2].plot(los_attn,  marker='o', color='blue')
        axes[2].plot(nlos_attn, marker='x', linestyle='--', color='orange')
        axes[2].set_title("latent_attention (64 dims)")
        axes[2].set_ylabel("Activation")

        # ── los_logits (scalar probability)
        axes[3].bar([0], [los_logits],    width=0.4, color='blue',   label='LOS')
        axes[3].bar([1], [nlos_logits],   width=0.4, color='orange', label='NLOS')
        axes[3].set_xlim(-0.5, 1.5)
        axes[3].set_xticks([0, 1])
        axes[3].set_xticklabels(['LOS', 'NLOS'])
        axes[3].set_title("los_logits (sigmoid output)")
        axes[3].set_ylabel("Probability")
        axes[3].legend()

        plt.xlabel("Latent‐dimension index (where applicable)")
        plt.tight_layout()
        plt.savefig(f"{save_plots}/probe_ae_{i}.png")
        plt.close()


def plot_all_histories(history, save_path):
    os.makedirs(save_path, exist_ok=True)

    # 1) Group keys by their metric name (suffix after train_/val_/test_)
    metric_groups = {}
    for key, values in history.items():
        if key.startswith("train_"):
            phase, base = "train", key[len("train_"):]
        elif key.startswith("val_"):
            phase, base = "val", key[len("val_"):]
        elif key.startswith("test_"):
            phase, base = "test", key[len("test_"):]
        else:
            # assume un-prefixed == training metric
            phase, base = "train", key

        metric_groups.setdefault(base, {})[phase] = values

    # 2) For each metric, plot train/val/test on the same figure
    for base, group in metric_groups.items():
        plt.figure()
        for phase in ("train", "val", "test"):
            if phase in group:
                plt.plot(group[phase], label=phase)
        plt.title(f"{base} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(base)
        plt.gca().grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        plt.gca().set_axisbelow(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{base}.png"))
        plt.close()





def _to_array(v):
    return np.asarray(list(v), dtype=float)

def _ema(y, alpha):
    if not (0 < alpha <= 1) or len(y) < 2:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.empty_like(y); s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
    return s

def _prettify(name: str) -> str:
    return name.replace("_", " ").title()

def _split_role(key: str):
    """
    Return (role, base_metric) from keys like:
      'loss' -> ('train','loss')
      'train_dom_acc' -> ('train','dom_acc')
      'val_dom_acc' -> ('val','dom_acc')
      'test_dom_acc' -> ('test','dom_acc')
    """
    lower = key.lower()
    mapping = [
        ("train", ("train_",)),
        ("val",   ("val_", "validation_")),
        ("test",  ("test_", "eval_", "evaluation_")),
    ]
    for role, prefixes in mapping:
        for p in prefixes:
            if lower.startswith(p):
                return role, key[len(p):]
    return "train", key  # bare metrics default to 'train'


def _safe(name: str) -> str:
    """Filesystem-safe name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)

def plot_history_dashboard(
    history: dict,
    save_path: str,
    smoothing: float = 0.0,   # 0 disables EMA
    dpi: int = 200,
    filename_prefix: str = "metric",
    show_markers: bool = True,
    marker_every: int = 1,    # 1 = every epoch
    marker_size: float = 4.5,
    hollow_markers: bool = True,
):
    """
    For each base metric, create ONE figure with train/val/test in it and save to disk.
    No combined dashboard is created.
    """
    os.makedirs(save_path, exist_ok=True)

    # Clean, professional defaults; no explicit colors.
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "font.size": 10,
    })

    # Group by base metric → roles {train,val,test}
    grouped, order = {}, []
    for k, v in history.items():
        role, base = _split_role(k)
        grouped.setdefault(base, {})[role] = _to_array(v)
        if base not in order:
            order.append(base)

    style_map = {
        "train": {"linestyle": "-",  "marker": "o"},
        "val":   {"linestyle": "--", "marker": "^"},
        "test":  {"linestyle": ":",  "marker": "s"},
    }
    role_order = ["train", "val", "test"]

    for base in order:
        roles = grouped.get(base, {})
        if not roles:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        for role in role_order:
            if role not in roles:
                continue
            y = roles[role]
            y = _ema(y, smoothing) if smoothing else y
            x = np.arange(1, len(y) + 1)

            marker_kwargs = {}
            if show_markers:
                marker_kwargs = dict(
                    marker=style_map[role]["marker"],
                    markevery=marker_every,
                    markersize=marker_size,
                    fillstyle="none" if hollow_markers else "full",
                    markeredgewidth=1.2,
                )

            ax.plot(
                x, y,
                linewidth=2.0,
                linestyle=style_map[role]["linestyle"],
                label=role,
                **marker_kwargs
            )

        ax.set_title(_prettify(base), fontsize=13,fontweight="bold", pad=8)
        ax.set_xlabel("Epoch",fontsize=13,fontweight="bold"); ax.set_ylabel("Value",fontsize=13,fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        if base.lower() in {"lr", "learning_rate"}:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(loc="best", fontsize=9)

        # annotate last value
        for line in ax.lines:
            if len(line.get_xdata()) == 0:
                continue
            x_last, y_last = line.get_xdata()[-1], line.get_ydata()[-1]
            ax.annotate(f"{y_last:.4g}", xy=(x_last, y_last),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, va="center")

        fig.tight_layout()
        safe_base = _safe(base)
        png_path = os.path.join(save_path, f"{filename_prefix}_{safe_base}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


 



def plot_predicted_vs_real(y_pred, y_real, save_path):
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(y_pred)), y_pred, color='blue', label='NLOS predicted')
        plt.scatter(np.arange(len(y_pred)), y_real, color='red', label='LOS real')
        plt.title('Predicted LOS/NLOS Probabilities')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Probability')
        plt.legend()
        plt.savefig(f"{save_path}/predicted_los_nlos_probabilities.png")
        plt.close()



def _extract_nlos_prob(raw_pred):
    """Return a 1D array of P(NLOS) in [0,1] regardless of model head."""
    p = np.asarray(raw_pred)
    if p.ndim == 2 and p.shape[1] == 2:       # softmax [P(LOS), P(NLOS)]
        p = p[:, 1]
    else:                                     # sigmoid (N,1) or (N,)
        p = p.reshape(-1)
    # If this ends up being P(LOS), flip automatically:
    # (mean prob on NLOS should be higher than on LOS)
    return p

def plot_confusion_matrix(classifier,dt_roles, CIRS, Y, h, title="Confusion Matrix"):
    save_path = h.get("save_plots")
    os.makedirs(save_path, exist_ok=True)
    thr = h.get("METRIC_THRESHOLD")
    

    
    for split in dt_roles:
        X      = CIRS[split][..., None]
        y_true = Y[split].astype(int)

        raw    = classifier.predict(X, verbose=0).squeeze()
        probs  = _extract_nlos_prob(raw)             # -> P(NLOS)
        preds  = (probs >= thr).astype(int)

        print(classification_report(y_true, preds, target_names=["LOS","NLOS"]))
        cm = confusion_matrix(y_true, preds)
        acc = np.trace(cm) / np.sum(cm)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["LOS","NLOS"], yticklabels=["LOS","NLOS"])
        plt.title(f"{title} — {split}")
        plt.xlabel(f"Predicted\nAccuracy: {acc:.2f}")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{title} {split}.png"); plt.close()


def evaluate_model_performance(probe_model,dt_roles, CIRS, LosLabels, h):
    save_plots=h.get("save_plots")
    os.makedirs(save_plots, exist_ok=True)
    for split in dt_roles:
        try:
            X_los  = CIRS[split][LosLabels[split]==0][..., None]
            X_nlos = CIRS[split][LosLabels[split]==1][..., None]
            X_all  = np.concatenate([X_los, X_nlos], axis=0)
            y_all  = np.concatenate([np.zeros(len(X_los)), np.ones(len(X_nlos))]).astype(int)

            raw    = probe_model.predict(X_all, verbose=0)
            probs  = _extract_nlos_prob(raw[-1])   # last output is your prob head

            # ---- hist (good for AUC intuition)
            plt.figure(figsize=(12,6))
            plt.hist(probs[y_all==0], bins=30, alpha=0.6, label="LOS")
            plt.hist(probs[y_all==1], bins=30, alpha=0.6, label="NLOS")
            plt.xlabel("Predicted P(NLOS)"); plt.ylabel("Count")
            plt.title(f"{split} set probability distributions")
            plt.legend(); plt.tight_layout()
            plt.savefig(f"{save_plots}/{split}_set_probabilities.png"); plt.close()

            # ---- scatter (index vs prob, colored by class)
            idx0 = np.where(y_all==0)[0]
            idx1 = np.where(y_all==1)[0]
            plt.figure(figsize=(12,6))
            plt.scatter(idx0, probs[idx0], s=12, alpha=0.6, label="LOS")
            plt.scatter(idx1, probs[idx1], s=12, alpha=0.6, label="NLOS")
            plt.axhline(0.5, ls="--", lw=1)
            plt.xlabel("Sample index"); plt.ylabel("Predicted P(NLOS)")
            plt.title(f"{split} set probability scatter")
            plt.legend(); plt.tight_layout()
            plt.savefig(f"{save_plots}/{split}_set_probabilities_scatter.png"); plt.close()

            # optional: verify confusion matrix matches threshold
            preds = (probs >= h["METRIC_THRESHOLD"]).astype(int)
        
            print(split, "CM @0.5:\n", confusion_matrix(y_all, preds))
        except:
            print("error with plots")


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.colors import Normalize


def plot_confusion_matrices_grid(
    classifier, CIRS, Y, h,splits,s_d, title="Confusion Matrix",
    suptitle_fs=18, panel_title_fs=14, label_fs=12, tick_fs=11, annot_fs=12, cbar_label_fs=12
):
    save_path = h.get("save_plots")
    os.makedirs(save_path, exist_ok=True)
    n_dm=s_d["num_dom"]


    cms, accs = {}, {}
    for split in splits:
        X      = CIRS[split][..., None]
        y_true = Y[split].astype(int)
        raw    = classifier.predict(X, verbose=0)
        #probs  = _extract_nlos_prob(raw)
        preds  = raw.argmax(axis=1)

        print(f"\n=== {split} ===")
        print(classification_report(y_true, preds, target_names=["LOS","NLOS"]))

        cm = confusion_matrix(y_true, preds)
        cms[split] = cm
        accs[split] = np.trace(cm) / np.sum(cm)

    # figure without a shared colorbar column
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    if n_dm==4:
        gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1,1], width_ratios=[1, 1])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
        ]
        ax_spare = fig.add_subplot(gs[2, 1])
        ax_spare.axis('off')
    elif n_dm==3:
        gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
           
        ]

    elif n_dm==2:
        gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
        ]
        ax_spare = fig.add_subplot(gs[1, 1])
        ax_spare.axis('off')
           
    for ax, split in zip(axes, splits):
        # per-panel autoscale; no vmin/vmax
        im = sns.heatmap(
            cms[split], ax=ax, annot=True, fmt="d", cmap="Blues",
            cbar=False,
            xticklabels=["LOS","NLOS"], yticklabels=["LOS","NLOS"],
            annot_kws={"fontsize": annot_fs, "fontweight": "bold"}
        )

        # per-ax colorbar (same style as your other code)
        cb = fig.colorbar(im.collections[0], ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Count", fontsize=cbar_label_fs, fontweight="bold")
        cb.ax.tick_params(labelsize=tick_fs)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight("bold")

        ax.set_title(split, fontsize=panel_title_fs, fontweight="bold")
        ax.set_xlabel(f"Predicted\nAccuracy: {accs[split]:.2f}",
                      fontsize=label_fs, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=label_fs, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    out_png = os.path.join(save_path, f"{title}_ALL.png")
    out_pdf = os.path.join(save_path, f"{title}_ALL.pdf")
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_regression_error_grid(
    reg_model,          # e.g. model_prediction_error(ae)
    CIRS,               # dict: split -> X (N, T) or (N, T, 1)
    ErrorLabels,        # dict: split -> y_true (N,)
    h,
    splits,
    s_d,
    title="Error Regression",
    suptitle_fs=18, panel_title_fs=14, label_fs=12, tick_fs=11
):
    """
    Create a grid of scatter plots: y_true vs y_pred for each split
    (e.g., TRAIN1, TRAIN2, ADAPTION, TEST), with MSE/MAE/R^2 in the title.

    Layout mimics plot_confusion_matrices_grid so it integrates nicely
    into your existing reporting.
    """
    save_path = h.get("save_plots")
    os.makedirs(save_path, exist_ok=True)
    n_dm = s_d["num_dom"]

    # ----- choose layout similar to plot_confusion_matrices_grid -----
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    if n_dm == 4:
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
        ]
        ax_spare = fig.add_subplot(gs[2, 1])
        ax_spare.axis("off")
    elif n_dm == 3:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
    elif n_dm == 2:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
        ]
        ax_spare = fig.add_subplot(gs[1, 1])
        ax_spare.axis("off")
    else:
        # fallback: one row
        gs = fig.add_gridspec(1, len(splits))
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(splits))]

    # ----- plot per split -----
    global_min = +np.inf
    global_max = -np.inf
    split_stats = {}

    # first pass: compute global min/max for shared axes
    for split in splits:
        X = np.asarray(CIRS[split])
        if X.ndim == 2:
            X = X[..., None]   # (N, T, 1) for model

        y_true = np.asarray(ErrorLabels[split]).astype(np.float32).squeeze()
        y_pred = reg_model.predict(X, verbose=0).astype(np.float32).squeeze()

        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        global_min = min(global_min, mn)
        global_max = max(global_max, mx)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        # guard R^2 when y_true is constant
        if len(np.unique(y_true)) > 1:
            r2 = r2_score(y_true, y_pred)
        else:
            r2 = np.nan

        split_stats[split] = dict(
            y_true=y_true, y_pred=y_pred,
            mse=mse, mae=mae, r2=r2
        )

    # small padding
    pad = 0.05 * (global_max - global_min) if global_max > global_min else 0.1
    lims = (global_min - pad, global_max + pad)

    # second pass: draw plots
    for ax, split in zip(axes, splits):
        stats = split_stats[split]
        y_true = stats["y_true"]
        y_pred = stats["y_pred"]
        mse = stats["mse"]
        mae = stats["mae"]
        r2  = stats["r2"]

        ax.scatter(y_true, y_pred, s=12, alpha=0.6)
        ax.plot(lims, lims, linestyle="--", linewidth=1.2)  # y=x
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(
            f"{split}\nMSE={mse:.3g}, MAE={mae:.3g}, R²={r2:.3g}",
            fontsize=panel_title_fs, fontweight="bold"
        )
        ax.set_xlabel("True error_ratio", fontsize=label_fs, fontweight="bold")
        ax.set_ylabel("Predicted error_ratio", fontsize=label_fs, fontweight="bold")
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    fig.suptitle(title, fontsize=suptitle_fs, fontweight="bold", y=0.99)
    out_png = os.path.join(save_path, f"{title.replace(' ', '_')}_ALL.png")
    out_pdf = os.path.join(save_path, f"{title.replace(' ', '_')}_ALL.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _to_1d_float(arr):
    a = np.asarray(arr, dtype=np.float32)
    return a.squeeze()


def plot_range_overlay(
    *,
    sensor_rng,
    real_rng,
    pred_error_ratio,
    save_dir,
    scale: float = 10.0,
    title: str = "Range Comparison",
    filename: str = "range_overlay.png",
    max_points: int = 5000,
):
    """Plot real range, predicted (corrected) range, and sensor range in one figure.

    Your project defines:
      error_ratio = (sensor_rng - real_rng) / (sensor_rng + eps) 
    So we invert (using sensor_rng in the denominator):
      pred_error = pred_error_ratio * sensor_rng / scale
      pred_range = sensor_rng - pred_error
    """
    from pathlib import Path

    sensor_rng = _to_1d_float(sensor_rng)
    real_rng = _to_1d_float(real_rng)
    pred_error_ratio = _to_1d_float(pred_error_ratio)

    n = int(min(len(sensor_rng), len(real_rng), len(pred_error_ratio)))
    sensor_rng = sensor_rng[:n]
    real_rng = real_rng[:n]
    pred_error_ratio = pred_error_ratio[:n]

    # light decimation for huge N (keeps figure readable)
    if max_points is not None and n > int(max_points):
        step = max(1, n // int(max_points))
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    pred_error = (pred_error_ratio * sensor_rng) / float(scale)
    pred_rng = sensor_rng - pred_error

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    # Scatter (shows density) + line (shows trend)
    plt.scatter(idx, real_rng[idx], s=8, alpha=0.35)
    plt.scatter(idx, pred_rng[idx], s=8, alpha=0.35)
    plt.scatter(idx, sensor_rng[idx], s=8, alpha=0.25)

    plt.plot(idx, real_rng[idx], linewidth=1.2, alpha=0.9, label="Real range (camera)")
    plt.plot(idx, pred_rng[idx], linewidth=1.2, alpha=0.9, label="Predicted range (corrected)")
    plt.plot(idx, sensor_rng[idx], linewidth=1.2, alpha=0.7, label="Sensor range")
    plt.xlabel("Sample index", fontsize=12, fontweight="bold")
    plt.ylabel("Range", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
    plt.gca().set_axisbelow(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_overlay(
    *,
    sensor_rng,
    real_rng,
    pred_error_ratio,
    save_dir,
    scale: float = 10.0,
    title: str = "Error Comparison",
    filename: str = "error_overlay.png",
    max_points: int = 5000,
):
    """Plot predicted error and real error in one figure.

    real_error = sensor_rng - real_rng
    pred_error = pred_error_ratio * sensor_rng / scale
    """
    from pathlib import Path

    sensor_rng = _to_1d_float(sensor_rng)
    real_rng = _to_1d_float(real_rng)
    pred_error_ratio = _to_1d_float(pred_error_ratio)

    n = int(min(len(sensor_rng), len(real_rng), len(pred_error_ratio)))
    sensor_rng = sensor_rng[:n]
    real_rng = real_rng[:n]
    pred_error_ratio = pred_error_ratio[:n]

    if max_points is not None and n > int(max_points):
        step = max(1, n // int(max_points))
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    real_error = sensor_rng - real_rng
    pred_error = (pred_error_ratio * sensor_rng) / float(scale)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    # Scatter (shows density) + line (shows trend)
    plt.scatter(idx, real_error[idx], s=8, alpha=0.35)
    plt.scatter(idx, pred_error[idx], s=8, alpha=0.35)

    plt.plot(idx, real_error[idx], linewidth=1.2, alpha=0.9, label="Real error (sensor - camera)")
    plt.plot(idx, pred_error[idx], linewidth=1.2, alpha=0.9, label="Predicted error")
    plt.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.5)
    plt.xlabel("Sample index", fontsize=12, fontweight="bold")
    plt.ylabel("Error", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
    plt.gca().set_axisbelow(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=200, bbox_inches="tight")
    plt.close()
