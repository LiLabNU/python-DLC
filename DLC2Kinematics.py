import os
import cv2
import pandas as pd
import numpy as np
import dlc2kinematics
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# === Folder containing .h5 and .mp4 files ===
folder_path = r"R:\PBS\LiPatel_Labs\Personal_Folders\Talia\Behavior\MedPC_Data\DOB\Day10"
click_cache_csv = os.path.join(folder_path, "sucrose_port_coords.csv")

# === Load / create cache for sucrose‑port clicks ===
if os.path.exists(click_cache_csv):
    click_cache = pd.read_csv(click_cache_csv, index_col=0).to_dict("index")
else:
    click_cache = {}

# --------------------------------------------------------------------------
# Helper: ask user to click the sucrose port on the first frame
# --------------------------------------------------------------------------
def prompt_sucrose_click(video_path: str, hx: float, hy: float, base: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"❌ Could not read first frame of {video_path}")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.scatter(hx, hy, color="red", s=40, label="Head")
    plt.title(f"Click sucrose port for {base}")
    plt.legend()
    click = plt.ginput(1, timeout=0)
    plt.close()
    if not click:
        raise ValueError("No click registered.")
    return click[0]  # (x, y)

# --------------------------------------------------------------------------
# Core routine: process one DLC H5 + matching video
# --------------------------------------------------------------------------
def process_h5_file(h5_path: str, video_path: str, base: str):
    out_csv = h5_path.replace(".h5", "_dlc2kinematics.csv")
    print(f"\n🧠 Processing {base}")

    # Load DLC table
    df, bodyparts, scorer = dlc2kinematics.load_data(h5_path)
    df.columns.set_names(['scorer', 'bodyparts', 'coords'], inplace=True)  # Changed to 'bodyparts' (plural)

    # Interpolate low‑likelihood points
    for bp in bodyparts:
        like = df[scorer][bp]["likelihood"]
        good = like >= 0.9
        for coord in ("x", "y"):
            series = df[scorer][bp][coord]
            if good.sum() > 1:
                df[scorer][bp][coord] = np.interp(np.arange(len(series)), np.flatnonzero(good), series[good])
        df[scorer][bp]["likelihood"] = 1.0

    # Get / cache sucrose‑port
    if base not in click_cache:
        hx, hy = df[scorer]["head"]["x"].iat[0], df[scorer]["head"]["y"].iat[0]
        px, py = prompt_sucrose_click(video_path, hx, hy, base)
        click_cache[base] = {"x": px, "y": py}
        pd.DataFrame(click_cache).T.to_csv(click_cache_csv)
        print("💾 Port cached.")
    else:
        px, py = click_cache[base]["x"], click_cache[base]["y"]
        print(f"✅ Cached port: ({px:.1f}, {py:.1f})")

    # Add sucroseport bodypart
    port_cols = pd.MultiIndex.from_tuples([
        (scorer, "sucroseport", "x"),
        (scorer, "sucroseport", "y"),
        (scorer, "sucroseport", "likelihood"),
    ], names=["scorer", "bodyparts", "coords"])  # Changed to 'bodyparts' (plural)
    port_df = pd.DataFrame({
        port_cols[0]: [px] * len(df),
        port_cols[1]: [py] * len(df),
        port_cols[2]: [1.0] * len(df)
    }, index=df.index)
    df = pd.concat([df, port_df], axis=1)

    # Add virtual_body bodypart
    vb_x = (df[scorer]["head"]["x"] + df[scorer]["nose"]["x"] + df[scorer]["body"]["x"]) / 3
    vb_y = (df[scorer]["head"]["y"] + df[scorer]["nose"]["y"] + df[scorer]["body"]["y"]) / 3
    vb_cols = pd.MultiIndex.from_tuples([
        (scorer, "virtual_body", "x"),
        (scorer, "virtual_body", "y"),
        (scorer, "virtual_body", "likelihood"),
    ], names=["scorer", "bodyparts", "coords"])  # Changed to 'bodyparts' (plural)
    vb_df = pd.DataFrame({
        vb_cols[0]: vb_x,
        vb_cols[1]: vb_y,
        vb_cols[2]: np.ones_like(vb_x)
    }, index=df.index)
    df = pd.concat([df, vb_df], axis=1)

    # 🔧 Ensure column names are correct before kinematics
    df.columns.set_names(["scorer", "bodyparts", "coords"], inplace=True)  # Changed to 'bodyparts' (plural)

    # Kinematics
    df_vel = dlc2kinematics.compute_velocity(df, bodyparts=["virtual_body"])
    df_acc = dlc2kinematics.compute_acceleration(df, bodyparts=["virtual_body"])
    df_speed = dlc2kinematics.compute_speed(df, bodyparts=["virtual_body"])

    # Virtual body position and velocity
    vb_x = df[scorer]["virtual_body"]["x"]
    vb_y = df[scorer]["virtual_body"]["y"]
    vb_vx = df_vel[scorer]["virtual_body"]["x"]
    vb_vy = df_vel[scorer]["virtual_body"]["y"]

    # Distance from virtual body to port
    dx = px - vb_x
    dy = py - vb_y
    dist_to_port = np.sqrt(dx**2 + dy**2)

    # Speed of virtual body toward port
    # Normalize direction vector
    ux = dx / dist_to_port
    uy = dy / dist_to_port
    speed_to_port = vb_vx * ux + vb_vy * uy

    # Body length (unchanged)
    body_len = np.sqrt((df[scorer]["head"]["x"] - df[scorer]["tailbase"]["x"])**2 +
                       (df[scorer]["head"]["y"] - df[scorer]["tailbase"]["y"])**2)

    # Flatten helpers
    def flat_xy(tbl, tag):
        tbl = tbl.loc[:, tbl.columns.get_level_values("coords").isin(["x", "y"])]
        tbl.columns = [f"{bp}_{tag}_{coord}" for (_, bp, coord) in tbl.columns]
        return tbl

    def flat_scalar(tbl, tag):
        tbl = tbl.loc[:, tbl.columns.get_level_values("coords") != "likelihood"]
        tbl.columns = [f"{bp}_{tag}" for (_, bp, _) in tbl.columns]
        return tbl

    df_out = pd.concat([
        flat_xy(df_vel, "vel"),
        flat_xy(df_acc, "acc"),
        flat_scalar(df_speed, "speed"),
        pd.DataFrame({"virtual_body_dist_to_port": dist_to_port}),
        pd.DataFrame({"virtual_body_speed_to_port": speed_to_port}),
        pd.DataFrame({"body_length": body_len}),
    ], axis=1)

    df_out.to_csv(out_csv, index=False)
    print(f"✅ Saved → {out_csv}")

# --------------------------------------------------------------------------
# Iterate over files and run
# --------------------------------------------------------------------------
for fname in os.listdir(folder_path):
    if fname.endswith("_filtered.h5") and "DLC_" in fname:
        base = fname.split("DLC_")[0]
        h5_path = os.path.join(folder_path, fname)
        mp4_path = os.path.join(folder_path, f"{base}.mp4")
        if os.path.exists(mp4_path):
            process_h5_file(h5_path, mp4_path, base)
        else:
            print(f"⚠️ Video missing for {fname}")
