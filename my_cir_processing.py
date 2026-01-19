import numpy as np
def cutting_cir(df):
    cir=df["CIR_amp"].to_numpy()
    fp_idx=df["sensor fp_idx"].to_numpy()
    cut_cir=[]


    for c,first_peak in zip(cir, fp_idx):
     
         cut_cir.append(c[first_peak-50: first_peak+100])
                
    return np.array(cut_cir)


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
            norm_2d = ((cir - row_min) / (row_max - row_min))

            # back to 1-D
            res.append(norm_2d)

        CIRS["role"].append(name)
        CIRS["data"].append(np.array(res))


        # 4) Stack back into an (N_rows, length) array
    return dict(zip(CIRS["role"], CIRS["data"]))
