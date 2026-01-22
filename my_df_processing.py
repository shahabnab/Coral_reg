import pandas as pd
import gdown
import os
import pickle
from my_cir_processing import cutting_cir
import tensorflow as tf
import numpy as np
from my_losses import *

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
def shuffle_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
   
    return (
        df.sample(frac=1.0, random_state=seed)  # permute rows

    )
# Toggle what the regression label should represent:
# True  -> regression label is error ratio = (sensor - camera) / sensor
# False -> regression label is raw error   = (sensor - camera)
USE_ERROR_RATIO: bool = True

def _label_mode() -> str:
    """
    Global switch (set in your main):
      os.environ["CORAL_LABEL_MODE"] = "ratio" | "error"
    """
    return os.getenv("CORAL_LABEL_MODE", "ratio").strip().lower()

def _use_error_ratio() -> bool:
    return _label_mode() != "error"

def reg_label_col(use_error_ratio: bool | None = None) -> str:
    if use_error_ratio is None:
        use_error_ratio = _use_error_ratio()
    return "error ratio" if use_error_ratio else "error"

""" def get_error(df, use_error_ratio: bool | None = None):
    real_rng   = df["camera rng"].values.astype(np.float32)
    sensor_rng = df["Sensor rng"].values.astype(np.float32)

    error = sensor_rng - real_rng
    eps = 1e-6
    ratio = error / (sensor_rng + eps)

    # keep both always (so evaluation can switch safely)
    df["error"] = error.astype(np.float32)
    df["error ratio"] = ratio.astype(np.float32)

    # keep your existing downstream expectations stable:
    # the "active" regression label is always in this column
    df["reg_label"] = df[reg_label_col(use_error_ratio)].astype(np.float32)

    # (optional) if you rely on this filter later
    df["error in (cm)"] = (df["error"] * 100.0).astype(np.float32)

    return df """
def get_error(df, use_error_ratio: bool | None = None):
    """
    Computes error metrics and ADDS them as columns to the DataFrame.
    MUST return the DataFrame (not an array) to preserve compatibility.
    """
    # 1. Identify columns safely
    real_rng = df["camera rng"].values.astype(np.float32)
    
    if "Sensor rng" in df.columns:
        sensor_rng = df["Sensor rng"].values.astype(np.float32)
    elif "sensor_rng" in df.columns:
        sensor_rng = df["sensor_rng"].values.astype(np.float32)
    else:
        raise KeyError(f"Sensor range column not found. Available: {df.columns}")

    # 2. Compute metrics
    diff = (sensor_rng - real_rng).astype(np.float32)
    # Avoid division by zero for ratio
    ratio = diff / (sensor_rng + 1e-6)
    ratio = ratio.astype(np.float32)

    # 3. Modify DataFrame IN-PLACE
    #    (We add all versions so downstream code can pick what it needs)
    df["error"] = diff
    df["error ratio"] = ratio
    df["error in (cm)"] = diff * 100.0  # Crucial: used for filtering TU in download_dataset

    # 4. RETURN THE DATAFRAME
    return df

def download_dataset(datasets_names,test_adaption_name,SEED):

    ############################## Downloading the datasets ##########################################
    if not os.path.exists("data"):
        os.makedirs("data")
    # Download TUall
    if not os.path.exists("data/TUall.pickle"):
        print("Downloading TUall dataset...")
        gdown.download(id="1R2GOGED6jzLU8I35lu5SHT1jlpCmqk7R", output="data/TUall.pickle", quiet=False)
    # Download IOT_PT1
    if not os.path.exists("data/IOT_PT1.pickle"):
        print("Downloading IOT_PT1 dataset...")
        gdown.download(id="1Xil7h9fHvaFGIE2nWpWUXOIAg1us9Wlo", output="data/IOT_PT1.pickle", quiet=False)
    if not os.path.exists("data/IOT_PT2.pickle"):
    # Download IOT_PT2
        print("Downloading IOT_PT2 dataset...")
        gdown.download(id="1mgOcYSp9BjOqDgp4GYiaQy7Au0pEEsxO", output="data/IOT_PT2.pickle", quiet=False)
    if not os.path.exists("data/Graz.pickle"):
        print("Downloading Graz dataset...")
        gdown.download(id="1jv67EErv9n2Cm-i9Psm-Y-JSV9dJqVKm", output="data/Graz.pickle", quiet=False)

    # Download Office
    if not os.path.exists("data/Office.pickle"):
        print("Downloading Office dataset...")
        gdown.download(id="1QhLyo9_4pSfvkXhyJ5DrC4P2qZcf9OdV", output="data/Office.pickle", quiet=False)
############################## using them in dataframes ##########################################

    with open("data/TUall.pickle", "rb") as f:
        TUall = pickle.load(f)
    with open("data/IOT_PT1.pickle", "rb") as f:
        IOT_PT1 = pickle.load(f)
    with open("data/IOT_PT2.pickle", "rb") as f:
        IOT_PT2 = pickle.load(f)
    with open("data/Office.pickle", "rb") as f:
        Office = pickle.load(f)
    with open("data/Graz.pickle", "rb") as f:
        Graz = pickle.load(f)    
    #concatenating the IOT_PT1 and IOT_PT2 datasets

    IOT = pd.concat([IOT_PT1[["label", "CIR_amp","Sensor rng","sensor rssi","sensor fp_power","camera rng"]], IOT_PT2[["label", "CIR_amp","Sensor rng","sensor rssi","sensor fp_power","camera rng"]]], axis=0, ignore_index=True).reset_index(drop=True)


    IOT.rename(columns={"label": "Label"}, inplace=True)
    IOT["Label"] = IOT["Label"].astype(int)
    #chanigng the LOS and NLOS labels to be 0 and 1 in TU dataset
    TUall["CIR_amp"] = TUall["CIR_amp"].apply(lambda x: np.sqrt(x) / 101)
    mask = TUall["sensor los"].isin({"los", "nlos"})
    TU = TUall.loc[mask].copy()
    # map string labels → ints on the copy
    TU["Label"] = TU["sensor los"].map({"los": 0, "nlos": 1}).astype(int)

    ###########################
    print("first path datatype:",Graz["sensor fp_idx"][0].dtype)

    Graz_cuts=cutting_cir(Graz)
    cuts_list_Graz = [row for row in Graz_cuts]
    Graz["CIR_amp"] = cuts_list_Graz 
    Graz.rename(columns={"sensor los": "Label"}, inplace=True)


    ##########################


    TU_cuts=cutting_cir(TU)
    cuts_list_TU = [row for row in TU_cuts]
    TU["CIR_amp"] = cuts_list_TU

    Office.rename(columns={"label": "Label"}, inplace=True)
    Office["Label"]=Office["Label"].astype(int)
    print("#################################################")
    print("#################################################")
    print("LOS and NLOS distribution in IOT dataset:",IOT["Label"].value_counts())
    print("#################################################")
    print("LOS and NLOS distribution in TU dataset:",TU["Label"].value_counts())
    print("#################################################")
    print("LOS and NLOS distribution in Office dataset:",Office["Label"].value_counts())
    print("#################################################")
    print("#################################################")
    IOT["Sensor rng"]=IOT["Sensor rng"]/100
    Office["Sensor rng"]=Office["Sensor rng"]/100
    IOT["camera rng"]=IOT["camera rng"]/100
    Office["camera rng"]=Office["camera rng"]/100
    

    IOT   = shuffle_df(IOT,   seed=SEED)
    TU   = shuffle_df(TU,   seed=SEED)
    Office = shuffle_df(Office, seed=SEED)
    new_IOT = get_error(IOT)
    new_TU = get_error(TU)
    new_Office = get_error(Office)
    new_TU=new_TU[(TU["error in (cm)"]<20)&(TU["error in (cm)"]>-20)]


    



    dfs = {
        "IOT": new_IOT,
        "TU": new_TU,
        "Office": new_Office,
       
    }
    

    datasets={}
    for id,name in enumerate(datasets_names):
         datasets[f"TRAIN{id+1}"] = dfs[name]
         print(f"TRAIN{id+1}:name:{name} shape:", datasets[f"TRAIN{id+1}"].shape)
    datasets["test_adaption"] = dfs[test_adaption_name]
    return datasets

def slicing_dts(datasets,s_d):
    datasets_names=s_d["Training_datasets"]
    dt_rules=s_d["DATASET_ROLES"]
    tr_size=s_d["train_size"]
    SEED=s_d["SEED"]
    test_adaption = datasets["test_adaption"].copy()

    # ── 1) convenience masks ───────────────────────────────────────────────
    def los(df):  return df[df["Label"] == 0]
    def nlos(df): return df[df["Label"] == 1]

    # ── 2) make balanced (LOS/NLOS) samples for every input split ─────────
    dts = {}
    for key, df in datasets.items():
        print(f"{key} dataset shape:", df.shape)

        nlos_s = nlos(df).sample(min(tr_size, len(nlos(df))), random_state=SEED)
        los_s  =  los(df).sample(min(tr_size, len( los(df))), random_state=SEED)

        dts[key] = pd.concat([nlos_s, los_s], ignore_index=True)
        dts[key] = shuffle_df(dts[key], seed=SEED)

    # report sizes
    for k, df in datasets.items():
        print(f"{k}: NLOS={len(nlos(df))}  LOS={len(los(df))}")

    # ── 3) ADAPTION  ──────────────────────────────────────────────────────
    ad_nlos = nlos(test_adaption).sample(min(tr_size, len(nlos(test_adaption))), random_state=SEED)
    ad_los  =  los(test_adaption).sample(min(tr_size, len( los(test_adaption))), random_state=SEED)
    ADAPTION = pd.concat([ad_nlos, ad_los])           # keep original indices for drop()
    TEST_REMAINED = test_adaption.drop(ADAPTION.index)

    # ── 4) TEST  ──────────────────────────────────────────────────────────
    test_nlos = nlos(TEST_REMAINED).sample(min(1000, len(nlos(TEST_REMAINED))), random_state=SEED)
    test_los  =  los(TEST_REMAINED).sample(min(1000, len( los(TEST_REMAINED))), random_state=SEED)
    TEST = pd.concat([test_nlos, test_los], ignore_index=True)

    ADAPTION = shuffle_df(ADAPTION, seed=SEED).reset_index(drop=True)
    TEST     = shuffle_df(TEST,     seed=SEED).reset_index(drop=True)

    keep_trains = s_d["Training_labels"]
    balanced_dts={}
    print(dts.keys())
    mx_train=0
    for i in keep_trains:
         balanced_dts[i]=dts[i].copy() 
         print(f"balanced_dts[{i}]:",balanced_dts[i].shape)
         mx_train+=1
    balanced_dts["ADAPTION"]=ADAPTION.copy()
    balanced_dts["TEST"]=TEST.copy() 
    balanced_dts["test_adaption"]=test_adaption.copy()
    print(f"ADAPTION adaption shape: {balanced_dts['ADAPTION'].shape}")
    print(f"TEST dataset shape: {balanced_dts['TEST'].shape}")


    return balanced_dts


def unpack_dataset(dts):
                X,R,D, L, W = [], [], [], [], []
                for x, (rec, dom, y_los,sw_los) in dts:
                    X.append(x.numpy())
                    R.append(rec.numpy())
                    D.append(dom.numpy())
                    L.append(y_los.numpy())
                    W.append(sw_los.numpy())
                X = np.concatenate(X, axis=0)
                R = np.concatenate(R, axis=0).squeeze()
                D = np.concatenate(D, axis=0).squeeze()
                L = np.concatenate(L, axis=0).squeeze()
                W = np.concatenate(W, axis=0).squeeze()
                return X,R,D,L,W

def take_reg_labels(dts, col: str | None = None, use_error_ratio: bool | None = None):
    """
    Default: uses df['reg_label'] if present; otherwise follows CORAL_LABEL_MODE.
    """
    if col is None:
        any_df = next(iter(dts.values()))
        col = "reg_label" if "reg_label" in any_df.columns else reg_label_col(use_error_ratio)

    out = {}
    for k, df in dts.items():
        out[k] = df[col].to_numpy(dtype=np.float32)
    return out
def make_dataset(
    X, d, l, w,
    h,_s_d,
    shuffle_buf=4096,
    split="train",
    pl_mode="hide",
):
    batch   = h["AE_BATCH"]
    num_dom = _s_d["num_dom"]
    seed    = _s_d["SEED"]
    pl_domain_id = _s_d["PL_DOMAIN_ID"]

    # --- host dtypes ---
    X = X.astype("float32")
    d = d.astype("int32")
    l = l.astype("float32")      # regression label (ratio OR raw error depending on pipeline)
    w = w.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((X, d, l, w))

    def _prep(sig, dom, lbl, w_los):
        # (T,) -> (T,1)
        if tf.rank(sig) == 1:
            sig = tf.expand_dims(sig, -1)

        sig   = tf.cast(sig,  tf.float32)
        y_rec = sig

        dom   = tf.cast(dom, tf.int32)
        y_dom = tf.one_hot(dom, depth=int(num_dom), dtype=tf.float32)

        
        y_los = tf.cast(lbl, tf.float32)
        y_los = tf.expand_dims(y_los, -1)   # (B,1) برای regression_loss و MSE metric

        sw_los = tf.cast(w_los, tf.float32)

        if split != "test" and pl_mode == "hide":
            is_target = tf.equal(dom, tf.cast(pl_domain_id, dom.dtype))
            sw_los = tf.where(is_target, 0.0, sw_los)

        return sig, (y_rec, y_dom, y_los, sw_los)

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch, drop_remainder=True)
    if split == "val" and pl_mode == "hide":
        ds = ds.filter(lambda sig, y: tf.reduce_any(y[3] != 0.0))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if split == "train":
        ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)

    return ds



