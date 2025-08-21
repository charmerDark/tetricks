import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV, treat "-" as NaN
df = pd.read_csv("mapper_data.csv", na_values="-")

# Convert runtimes to numeric
df["CGRA_Mapper"] = pd.to_numeric(df["CGRA_Mapper"], errors="coerce")
df["My_Mapper"]   = pd.to_numeric(df["My_Mapper"], errors="coerce")

# Kernel names
kernels = df["Kernel"]
cgra = df["CGRA_Mapper"]
my = df["My_Mapper"]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Bar positions
x = np.arange(len(kernels))
width = 0.35

# Bars (orange + blue)
bars1 = ax.bar(x - width/2, cgra, width, label="CGRA-Mapper", color="tab:orange")
bars2 = ax.bar(x + width/2, my, width, label="Proposed Mapper", color="tab:blue")

# Axis labels and title
ax.set_xlabel("Kernel", fontsize=16)
ax.set_ylabel("Runtime (cycles)", fontsize=16)
ax.set_title("Initiation Interval Comparison: CGRA-Mapper vs Proposed Mapper", fontsize=18, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=16)
ax.legend(fontsize=14)

# Annotate missing CGRA values with red X
for i, val in enumerate(cgra):
    if pd.isna(val):
        ax.annotate("X", xy=(i - width/2, 0), ha="center", va="bottom", fontsize=18, color="red")

# Annotate percentage improvement above My Mapper bars
for i, (cgra_val, my_val) in enumerate(zip(cgra, my)):
    if not pd.isna(cgra_val) and my_val > 0:
        perc_improve = ((cgra_val - my_val) / cgra_val) * 100
        ax.annotate(f"{perc_improve:.1f}%",
                    xy=(i + width/2, my_val),
                    xytext=(0, 5),  # offset above bar
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="green")

# Improve layout
plt.tight_layout()
plt.grid(axis="y", alpha=0.3)
plt.show()
