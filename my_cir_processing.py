import numpy as np
def CIR_pipeline(dfs):
    CIRS = {"role": [], "data": []}
    
    # 1. Identify Training Data to calculate the Reference Max
    #    (Assuming keys like 'TRAIN1', 'TRAIN2' exist)
    train_arrays = []
    for name, df in dfs.items():
        if "TRAIN" in name: 
            # Flatten to find global max easily
            # We use absolute because CIR is complex magnitude
            raw_amps = np.concatenate(df["CIR_amp"].to_numpy())
            train_arrays.append(np.abs(raw_amps))
            
    # Calculate the Fixed Reference (The "Ruler")
    if train_arrays:
        combined_train = np.concatenate(train_arrays)
        # Add a small buffer (e.g., 10%) to keep most values < 1.0
        GLOBAL_MAX_AMP = np.max(combined_train) 
    else:
        # Fallback if no train keys found (e.g. inference mode)
        GLOBAL_MAX_AMP = 6000.0 
        
    print(f"--- PHYSICS INFO ---")
    print(f"Global Reference Amplitude found: {GLOBAL_MAX_AMP}")
    print(f"Applying Log-Scaling: log(1+|x|) / log(1+{GLOBAL_MAX_AMP})")

    # 2. Apply this ONE constant to ALL data (Train & Test)
    log_denominator = np.log1p(GLOBAL_MAX_AMP)
    
    for name, df in dfs.items():
        raw_data = np.vstack(df["CIR_amp"].values).astype(np.float32)
        
        # Log-scale to compress noise but keep relative strength
        # Note: We use the SAME log_denominator for everyone
        norm_data = np.log1p(np.abs(raw_data)) / log_denominator
        
        CIRS["role"].append(name)
        CIRS["data"].append(norm_data)

    return dict(zip(CIRS["role"], CIRS["data"]))


#Reading the CIRS from the dataframes
def CIR_pipeline(dfs):
    CIRS={
        "role": [],
        "data": [],

    }
    for name in  (dfs.keys()):
        temp_cir=[]
        df= dfs[name].copy()
        res=[]
        print(f"index {name}")
        temp_cir = df["CIR_amp"].to_numpy()
        for cir in temp_cir:
             


            # fit_transform expects shape (n_samples, n_features)
            row_min = np.min(cir)
            row_max = np.max(cir)
            #norm_2d = ((cir - row_min) / (row_max - row_min))
            norm_2d = np.log1p(np.abs(cir)) / np.log1p(GLOBAL_MAX_AMP)

            # back to 1-D
            res.append(norm_2d)

        CIRS["role"].append(name)
        CIRS["data"].append(np.array(res))


        # 4) Stack back into an (N_rows, length) array
    return dict(zip(CIRS["role"], CIRS["data"]))
