import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# === SET PATHS ===
video_path = r"R:\PBS\LiPatel_Labs\Personal_Folders\Talia\Behavior\MedPC_Data\Pavlov_pilot2\Day11\New folder\BOX4-20241123T153427-170831.mp4"
h5_path = r"R:\PBS\LiPatel_Labs\Personal_Folders\Talia\Behavior\MedPC_Data\Pavlov_pilot2\Day11\New folder\BOX4-20241123T153427-170831DLC_resnet50_PavlovianOct11shuffle1_300000_filtered.h5"

# 1. Check file accessibility
if not os.path.isfile(h5_path):
    raise FileNotFoundError(f"H5 not found: {h5_path}")

print("Write access?", os.access(h5_path, os.W_OK))  # Should be True if writable

# 2. Load first frame of video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
if not ret:
    raise ValueError("Failed to read first video frame.")

# 3. Load the DLC data, print original sucroseport for first 5 frames
df = pd.read_hdf(h5_path, key='df_with_missing')
scorer = df.columns.levels[0][0]
print("\n=== Original Sucroseport (first 5 frames) ===")
print(df[scorer]['sucroseport'].head(5))

# 4. Show head location and wait for user click
head_x = df[scorer]['head']['x'].iloc[0]
head_y = df[scorer]['head']['y'].iloc[0]

plt.figure()
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.scatter(head_x, head_y, color='red', s=50, label='Head')
plt.title("Click on sucrose port (window will close after 1 click)")
plt.legend()
click = plt.ginput(1, timeout=0)
plt.close()

if not click:
    raise ValueError("No click registered.")

suc_x, suc_y = click[0]
print(f"Clicked => (x={suc_x:.1f}, y={suc_y:.1f})")

# 5. Update sucroseport for ALL frames
df[scorer]['sucroseport']['x'] = suc_x
df[scorer]['sucroseport']['y'] = suc_y
df[scorer]['sucroseport']['likelihood'] = 1.0

df.to_hdf(h5_path, key='df_with_missing', mode='w')
print("✅ Overwrote H5 with new sucroseport.")

# 6. Reload the H5 to confirm changes
df_check = pd.read_hdf(h5_path, key='df_with_missing')
print("\n=== Updated Sucroseport (first 5 frames) ===")
print(df_check[scorer]['sucroseport'].head(5))
