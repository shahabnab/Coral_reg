
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.losses import MeanSquaredError as mse_loss
#from my_models import *
from my_losses import *
from my_df_processing import *
def prob_class1(y):  # works for (N,2) softmax or (N,) / (N,1)
    #y = np.asarray(y)
     return y.argmax(axis=-1).astype(int)
    #return y[:, 1] if (y.ndim == 2 and y.shape[-1] == 2) else y.squeeze()

def argmax_labels(y_soft):  # (N,2) -> (N,)
    preds=prob_class1(y_soft)
    return preds

def set_seed(seed: int, *, enable_tf_op_determinism: bool = True, verbose: bool = True) -> None:
    """
    Reproducible runs on GPU without disabling parallelism.

    What it does:
      • Seeds Python, NumPy, and TensorFlow RNGs.
      • (Optionally) enables deterministic TF ops (cuDNN) when available.
      • DOES NOT force thread counts to 1 (keeps performance).

    Tips (do these in your entry script BEFORE importing TensorFlow):
      os.environ["TF_DETERMINISTIC_OPS"] = "1"
      os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
      # optional:
      # os.environ["PYTHONHASHSEED"] = str(seed)  # only effective at process start

    For tf.data pipelines, keep order deterministic:
      ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=False)
      ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    """
    import random
    import numpy as np
    import tensorflow as tf

    # RNGs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prefer TF's built-in deterministic switch (TF ≥ 2.12)
    if enable_tf_op_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # older TF versions: rely on the env vars set before import
            pass

    # Do NOT force single-thread execution — keep resources available.
    # If you ever want capped-but-parallel threads, set fixed numbers, e.g.:
    # tf.config.threading.set_intra_op_parallelism_threads(4)
    # tf.config.threading.set_inter_op_parallelism_threads(2)

    if verbose:
        print(f"[seed] {seed}  | TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

def take_sensors_ranges(datasets):
    
    
    dts_lbs={}
    for i in datasets.keys():
        dts_lbs[i] = datasets[i]["camera rng"].astype(np.float32)
    Labels = {
        "role": [],
        "data": []
    }
    for i in dts_lbs.keys():
        Labels["role"].append(i)
        Labels["data"].append(dts_lbs[i])
    cm_rng= dict(zip(Labels["role"], Labels["data"]))   

    dts_lbs={}
    for i in datasets.keys():
        dts_lbs[i] = datasets[i]["Sensor rng"].astype(np.float32)
    Labels = {
        "role": [],
        "data": []
    }
    for i in dts_lbs.keys():
        Labels["role"].append(i)
        Labels["data"].append(dts_lbs[i])
    sensor_rng= dict(zip(Labels["role"], Labels["data"]))  
    return  sensor_rng, cm_rng

def take_labels(datasets):
    dts_lbs={}
    for i in datasets.keys():
        dts_lbs[i] = datasets[i]["error ratio"].astype(np.float32)
    Labels = {
        "role": [],
        "data": []
    }
    for i in dts_lbs.keys():
        Labels["role"].append(i)
        Labels["data"].append(dts_lbs[i])
    return  dict(zip(Labels["role"], Labels["data"]))

def take_domains(datasets):
    Domains={
        "role": [],
        "data": []
    }
    
    for idx,i in enumerate(datasets.keys()):
        print(f"dataset {i} shape:", datasets[i].shape)
        Domains["role"].append(i)
        if i=="TEST":
             Domains["data"].append(np.full(len(datasets[i]), idx-1, dtype=np.int32))
        elif i=="test_adaption":
             Domains["data"].append(np.full(len(datasets[i]), idx-2, dtype=np.int32))     
        else:
             Domains["data"].append(np.full(len(datasets[i]), idx, dtype=np.int32))

    return dict(zip(Domains["role"], Domains["data"]))

def take_weights(datasets,adapt_size=0):
    weights={
        "role": [],
        "data": []
    }
    
    for idx,i in enumerate(datasets.keys()):
        print(f"dataset {i} shape:", datasets[i].shape)
        weights["role"].append(i)
        if i=="ADAPTION":
             temp=np.zeros(len(datasets[i]), dtype=np.float32)

             if adapt_size > 0:
                idx0=np.where(datasets["ADAPTION"]["Label"]==0)[0]
                idx1=np.where(datasets["ADAPTION"]["Label"]==1)[0]
                chosen0 = np.random.choice(idx0, size=adapt_size, replace=False)
                chosen1 = np.random.choice(idx1, size=adapt_size, replace=False)
                temp[chosen0] = 1
                temp[chosen1] = 1 
             weights["data"].append(temp)
        else:
             weights["data"].append(np.ones(len(datasets[i]), dtype=np.float32))

    return dict(zip(weights["role"], weights["data"]))



def train_valid_split(N,train_ratio,random_seed):
    np.random.seed(random_seed)
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_size = int(train_ratio * N)  # 80% for training
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    return train_indices, valid_indices

def print_distribution(data_dis):
    unique, counts = np.unique(data_dis, return_counts=True)
    for val, count in zip(unique, counts):
            print(f"{val}: {count}")

def save_to_excel(config, excel_path):
    



    df_cfg = pd.DataFrame([config])
    save_path=excel_path/"report.xlsx"
    df_cfg.to_excel(save_path, index=False)
    print(f"Configuration written to {save_path}")


""" def predict_los_only(
        
    CIRS,
    labels,
    Domains,
    Weights,
    lb_role: str,
    ae,
    h: dict,
  
):
    los_model = model_prediction_los(ae)
    dom_model = model_prediction_domain(ae)
    rec_model = model_prediction_reconstruction(ae)
    if "Train" in lb_role:
    
    
        X = CIRS[lb_role].copy()
        L = labels[lb_role].copy()
        W= Weights[lb_role].copy()
        D= Domains[lb_role].copy()
        num_dom=len(set(D.tolist()))
        D = tf.one_hot(D, depth=int(num_dom), dtype=tf.float32)
        
        mask_labeled = (W != 0)
        mask_unlabeled = (W == 0)
        los_probs = los_model.predict(X, verbose=0).squeeze()
        preds_bin=los_probs.argmax(axis=1)
        rec=rec_model.predict(X, verbose=0)
        dom=dom_model.predict(X, verbose=0).squeeze()
        sig = tf.expand_dims(X, -1)
        sig_labeled    = sig[mask_labeled]
        sig_unlabeled  = sig[mask_unlabeled]
        #labeled evaluation

        L_labeled    = L[mask_labeled]
        preds_labeled = preds_bin[mask_labeled]
        rec_labeled  = rec[mask_labeled]
        
        #unlabeled evaluation
        L_unlabeled    = L[mask_unlabeled]
        preds_unlabeled = preds_bin[mask_unlabeled]
        rec_unlabeled  = rec[mask_unlabeled]
    

        accuracy_unlabeled = accuracy_score(L_unlabeled, preds_unlabeled)
        f1_unlabeled       = f1_score    (L_unlabeled, preds_unlabeled)
        cm_unlabeled       = confusion_matrix(L_unlabeled, preds_unlabeled)

        accuracy_labeled = accuracy_score(L_labeled, preds_labeled)
        f1_labeled       = f1_score    (L_labeled, preds_labeled)
        cm_labeled       = confusion_matrix(L_labeled, preds_labeled)

        accuracy_all = accuracy_score(L, preds_bin)
        f1_all       = f1_score    (L, preds_bin)
        cm_all       = confusion_matrix(L, preds_bin)

        ce_focal  = CategoricalFocalCrossentropy(gamma=h["FOCAL_GAMMA"],from_logits=False) 
        dm_loss=ce_focal(dom, D)
        
        mse_loss_all=reconstruction_loss(sig, rec).numpy()
        mse_loss_labeled=reconstruction_loss(sig_labeled, rec_labeled).numpy()
        mse_loss_unlabeled=reconstruction_loss(sig_unlabeled, rec_unlabeled).numpy()
        res={
            f"{lb_role}_accuracy_labeled": accuracy_labeled,
            f"{lb_role}_f1_labeled": f1_labeled,
            f"{lb_role}_confusion_matrix_labeled": cm_labeled,
            f"{lb_role}_accuracy_unlabeled": accuracy_unlabeled,
            f"{lb_role}_f1_unlabeled": f1_unlabeled,
            f"{lb_role}_confusion_matrix_unlabeled": cm_unlabeled,
            f"{lb_role}_accuracy_all": accuracy_all,
            f"{lb_role}_f1_all": f1_all,
            f"{lb_role}_confusion_matrix_all": cm_all,
            f"{lb_role}_reconstruction_loss_all": mse_loss_all,
            f"{lb_role}_reconstruction_loss_labeled": mse_loss_labeled,
            f"{lb_role}_reconstruction_loss_unlabeled": mse_loss_unlabeled,
            f"{lb_role}_domain_loss": dm_loss.numpy(),

        }

    return res """

def predict_los_only(
        
    CIRS,
    labels,
    Domains,
    Weights,
    lb_role: str,
    ae,
    h: dict,
  
):
    los_model = model_prediction_los(ae)
    dom_model = model_prediction_domain(ae)
    rec_model = model_prediction_reconstruction(ae)
    enable_label=False
    if lb_role== "ADAPTION4":
        enable_label=True

    X = CIRS[lb_role].copy()
    L = labels[lb_role].copy()
    W= Weights[lb_role].copy()
    D= Domains[lb_role].copy()
    num_dom=len(set(D.tolist()))
    D_ids = Domains[lb_role].copy()   # (N,) integer domain IDs
    D = tf.one_hot(D, depth=int(num_dom), dtype=tf.float32)
    
    los_probs = los_model.predict(X, verbose=0).squeeze()
    preds_bin=los_probs.argmax(axis=1)
    rec=rec_model.predict(X, verbose=0)
    dom=dom_model.predict(X, verbose=0).squeeze()
    sig = tf.expand_dims(X, -1)

    if enable_label:
        mask_labeled = (W != 0)
        mask_unlabeled = (W == 0)
        sig_labeled    = sig[mask_labeled]
        sig_unlabeled  = sig[mask_unlabeled]
        L_labeled    = L[mask_labeled]
        preds_labeled = preds_bin[mask_labeled]
        rec_labeled  = rec[mask_labeled]
        L_unlabeled    = L[mask_unlabeled]
        preds_unlabeled = preds_bin[mask_unlabeled]
        rec_unlabeled  = rec[mask_unlabeled]
        accuracy_unlabeled = accuracy_score(L_unlabeled, preds_unlabeled)
        f1_unlabeled       = f1_score    (L_unlabeled, preds_unlabeled)
        cm_unlabeled       = confusion_matrix(L_unlabeled, preds_unlabeled)
        accuracy_labeled = accuracy_score(L_labeled, preds_labeled)
        f1_labeled       = f1_score    (L_labeled, preds_labeled)
        cm_labeled       = confusion_matrix(L_labeled, preds_labeled)
        mse_loss_labeled=reconstruction_loss(sig_labeled, rec_labeled).numpy()
        mse_loss_unlabeled=reconstruction_loss(sig_unlabeled, rec_unlabeled).numpy()
        dom_accuracy = accuracy_score(
        np.asarray(D_ids).astype(int),
        np.argmax(dom, axis=1).astype(int)
    )
        res={
        f"{lb_role}_accuracy_labeled": accuracy_labeled,
        f"{lb_role}_f1_labeled": f1_labeled,
        f"{lb_role}_confusion_matrix_labeled": cm_labeled,
        f"{lb_role}_accuracy_unlabeled": accuracy_unlabeled,
        f"{lb_role}_f1_unlabeled": f1_unlabeled,
        f"{lb_role}_confusion_matrix_unlabeled": cm_unlabeled,
        f"{lb_role}_accuracy_all": accuracy_all,
        f"{lb_role}_f1_all": f1_all,
        f"{lb_role}_confusion_matrix_all": cm_all,
        f"{lb_role}_reconstruction_loss_all": mse_loss_all,
        f"{lb_role}_reconstruction_loss_labeled": mse_loss_labeled,
        f"{lb_role}_reconstruction_loss_unlabeled": mse_loss_unlabeled,
        f"{lb_role}_domain_loss": dm_loss.numpy(),
        f"{lb_role}_domain_accuracy": dom_accuracy,
        }
    else:
        res={}
        accuracy_all = accuracy_score(L, preds_bin)
        f1_all       = f1_score    (L, preds_bin)
        cm_all       = confusion_matrix(L, preds_bin)

        ce_focal  = CategoricalFocalCrossentropy(gamma=h["FOCAL_GAMMA"],from_logits=False) 
       # dm_loss=ce_focal(dom, D)
        
        mse_loss_all=reconstruction_loss(sig, rec).numpy()
        """ dom_accuracy = accuracy_score(
        np.asarray(D_ids).astype(int),
        np.argmax(dom, axis=1).astype(int)
    ) """
        
        res={
            
            f"{lb_role}_accuracy_all": accuracy_all,
            f"{lb_role}_f1_all": f1_all,
            f"{lb_role}_confusion_matrix_all": cm_all,
            f"{lb_role}_reconstruction_loss_all": mse_loss_all,
            #f"{lb_role}_domain_loss": dm_loss.numpy(),
            #f"{lb_role}_domain_accuracy": dom_accuracy,

        }
    

    return res
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_dataset(ae, df, h, Title=""):
    """
    Evaluate model on a tf.data.Dataset `df` for REGRESSION head (error_ratio):

      - error head metrics: MSE / MAE / R² (all, labeled, unlabeled)
      - reconstruction loss (all, labeled, unlabeled)
      - domain loss + domain accuracy
    """
    # Models
    error_model = model_prediction_error(ae)          # outputs error_ratio in [-1,1]
    dom_model   = model_prediction_domain(ae)         # domain softmax
    rec_model   = model_prediction_reconstruction(ae) # reconstructed signal

    # Extract arrays
    X, _, D, L, W = unpack_dataset(df)   # check unpack_dataset: (X, y_rec, D, y_task, W)
    L = np.asarray(L, dtype=np.float32).squeeze()   # regression labels
    W = np.asarray(W, dtype=np.float32).squeeze()

    # Masks for labeled/unlabeled
    mask_labeled   = (W != 0)
    mask_unlabeled = (W == 0)

    # Predictions
    y_pred = error_model.predict(X, verbose=0).squeeze()
    rec    = rec_model.predict(X, verbose=0)
    dom    = dom_model.predict(X, verbose=0)

    # Split subsets
    X_labeled        = X[mask_labeled]
    X_unlabeled      = X[mask_unlabeled]
    L_labeled        = L[mask_labeled]
    L_unlabeled      = L[mask_unlabeled]
    y_pred_labeled   = y_pred[mask_labeled]
    y_pred_unlabeled = y_pred[mask_unlabeled]
    rec_labeled      = rec[mask_labeled]
    rec_unlabeled    = rec[mask_unlabeled]

    # --- Regression metrics ---
    mse_all = mean_squared_error(L, y_pred)
    mae_all = mean_absolute_error(L, y_pred)
    r2_all  = r2_score(L, y_pred)

    mse_labeled = mean_squared_error(L_labeled, y_pred_labeled) if mask_labeled.any() else np.nan
    mae_labeled = mean_absolute_error(L_labeled, y_pred_labeled) if mask_labeled.any() else np.nan
    r2_labeled  = r2_score(L_labeled, y_pred_labeled) if mask_labeled.any() and len(np.unique(L_labeled)) > 1 else np.nan

    mse_unlabeled = mean_squared_error(L_unlabeled, y_pred_unlabeled) if mask_unlabeled.any() else np.nan
    mae_unlabeled = mean_absolute_error(L_unlabeled, y_pred_unlabeled) if mask_unlabeled.any() else np.nan
    r2_unlabeled  = r2_score(L_unlabeled, y_pred_unlabeled) if mask_unlabeled.any() and len(np.unique(L_unlabeled)) > 1 else np.nan

    # --- Reconstruction losses (MSE) ---
    mse_loss_all        = reconstruction_loss(X, rec).numpy()
    mse_loss_labeled    = reconstruction_loss(X_labeled, rec_labeled).numpy() if mask_labeled.any() else np.nan
    mse_loss_unlabeled  = reconstruction_loss(X_unlabeled, rec_unlabeled).numpy() if mask_unlabeled.any() else np.nan

    # --- Domain metrics ---
    true_dom_ids = np.argmax(D, axis=1).astype(int)
    pred_dom_ids = np.argmax(dom, axis=1).astype(int)

    dom_accuracy = accuracy_score(true_dom_ids, pred_dom_ids)

    # domain loss (no CDAN-E weighting; just CE mean)
    D_tf   = tf.convert_to_tensor(D, tf.float32)
    dom_tf = tf.convert_to_tensor(dom, tf.float32)
    dm_loss = domain_loss_without_cdane(D_tf, dom_tf, h).numpy()

    res = {
        # regression metrics
        f"{Title}_mse_all": mse_all,
        f"{Title}_mae_all": mae_all,
        f"{Title}_r2_all":  r2_all,
        f"{Title}_mse_labeled": mse_labeled,
        f"{Title}_mae_labeled": mae_labeled,
        f"{Title}_r2_labeled":  r2_labeled,
        f"{Title}_mse_unlabeled": mse_unlabeled,
        f"{Title}_mae_unlabeled": mae_unlabeled,
        f"{Title}_r2_unlabeled":  r2_unlabeled,

        # reconstruction
        f"{Title}_reconstruction_loss_all":        mse_loss_all,
        f"{Title}_reconstruction_loss_labeled":    mse_loss_labeled,
        f"{Title}_reconstruction_loss_unlabeled":  mse_loss_unlabeled,

        # domain
        f"{Title}_domain_loss":     dm_loss,
        f"{Title}_domain_accuracy": dom_accuracy,
    }

    return res



   
