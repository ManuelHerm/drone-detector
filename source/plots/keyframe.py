from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

mpl.rcParams["figure.dpi"] = 600
plt.style.use("seaborn-v0_8")
plt.rcParams.update(
    {
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
    }
)

x = pd.Series([0, 0, 50, 50], index=[1, 20, 80, 100])

deg_frame = (0, 360)
deg = []
frames = []
current_frame = 0
step_size = 5
frame_steps = (step_size, 0)
for i in range(0, 40):
    deg.append(deg_frame[i % 2])
    frames.append(current_frame)
    current_frame += frame_steps[i % 2]

deg = pd.Series(deg, index=frames)

keyframes_deg = pd.Series(deg_frame, index=[0, step_size])
keyframes_x = pd.Series((0, 50), index=(20, 80))

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)

ax1.plot(x, color="C0", label="Drone Position")
ax1.set_ylabel("Drone Position [m]")
ax1.plot(keyframes_x, color="C0", label="Position Keyframes", linestyle="", marker="o")
ax1.legend()

ax2.plot(deg, color="C1", label="Propeller Position")
ax2.set_ylabel("Rotational Position [°]")
ax2.set_yticks([i * 90 for i in range(0, 5)])
ax2.set_xticks([i * 10 for i in range(0, 11)])
ax2.plot(
    keyframes_deg, color="C1", label="Rotation Keyframes", linestyle="", marker="o"
)
ax2.legend()
plt.tight_layout()
plt.savefig("keyframe.svg")
