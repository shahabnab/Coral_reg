#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import os

WANTED_COLS = ["Train1", "Train2", "Train3", "AE_BATCH", "AE_EPOCHS", "Test", "test_adaption_mae_sensor_vs_camera","test_adaption_mae_pred_vs_camera","test_adaption_improvement_ratio_mae"
,"test_adaption_mse_sensor_vs_camera","test_adaption_mse_pred_vs_camera","test_adaption_improvement_ratio_mse",
"TEST_mse_sensor_vs_camera","TEST_mse_pred_vs_camera","TEST_improvement_ratio_mse","TEST_mae_sensor_vs_camera","TEST_mae_pred_vs_camera","TEST_improvement_ratio_mae"

]




def collect_reports():
    """
    Walk `root`, find directories starting with 'Final', read 'reports.xlsx'
    (first sheet by default), and vertically concatenate the selected columns.
    """
    p=Path(".")
    folders=[f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f))]
    frames = []
    for final_dir in folders:
        final_dir = p / final_dir
        subfolders = [f for f in os.listdir(final_dir) if os.path.isdir(os.path.join(final_dir, f)) and f.startswith("FINAL")]
        for subfolder in subfolders:
            subfolder_path = final_dir / subfolder
            xlsx_path = subfolder_path / "report.xlsx"
            if xlsx_path.is_file():
                try:
                    df = pd.read_excel(xlsx_path, engine="openpyxl")
                    sel = df[WANTED_COLS].copy()
                    frames.append(sel)
                except Exception as e:
                    print(f"⚠️  Skipped {xlsx_path}: {e}")
        
    
    return pd.concat(frames, ignore_index=True)



combined = collect_reports()
combined.to_excel("all_reports.xlsx", index=False, engine="openpyxl")

    
