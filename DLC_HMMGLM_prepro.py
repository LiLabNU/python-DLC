import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

# === Folder containing *_dlc2kinematics.csv files ===
folder_path = r"R:\PBS\LiPatel_Labs\Personal_Folders\Talia\Behavior\MedPC_Data\Pavlov_pilot2\Day11"
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith("_dlc2kinematics.csv")])

# === Load and concatenate all sessions ===
all_df = []
session_ids = []

for i, file in enumerate(csv_files):
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # Drop raw x/y coordinates if not needed
    drop_cols = [col for col in df.columns if "_x" in col or "_y" in col]
    df = df.drop(columns=drop_cols)

    all_df.append(df)
    session_ids.extend([i] * len(df))

    print(f"📄 Loaded: {file} | {df.shape[0]} frames")

# === Combine all sessions into one long DataFrame ===
df_concat = pd.concat(all_df, ignore_index=True)

# === Z-score across the entire dataset (column-wise)
X_zscored = df_concat.apply(zscore, nan_policy='omit')

# === Final output
X = X_zscored.values
session_ids = np.array(session_ids)

# === Save output
np.save(os.path.join(folder_path, "hmmglm_X.npy"), X)
np.save(os.path.join(folder_path, "hmmglm_session_ids.npy"), session_ids)

print(f"\n✅ Saved HMMGLM input matrix: {X.shape}")
print(f"🧠 Session index vector: {len(session_ids)} entries")
