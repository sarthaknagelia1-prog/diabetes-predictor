import sys
sys.stdout.reconfigure(encoding="utf-8")   # Fix Windows cp1252 UnicodeEncodeError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer

# Colour constants
BLUE    = "#185FA5"
CORAL   = "#D85A30"
GRAY    = "#888780"
PALETTE = [BLUE, CORAL]

# -------------------------------------------------------------

print("=" * 60)
print("  DIABETES PREDICTION PIPELINE")
print("=" * 60)

df = pd.read_csv(r"C:\Users\sarth\Downloads\diabetes.csv")
print(f"\n[1] Data loaded — {df.shape[0]} rows × {df.shape[1]} cols")
print(df.head())

print("\n" + "-" * 60)
print("[2] DATA CLEANING")
print("-" * 60)

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero counts before cleaning (= hidden nulls):")
for col in ZERO_COLS:
    n = (df[col] == 0).sum()
    print(f"   {col:<25} {n:>4} rows  ({n/len(df)*100:.1f}%)")

df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=5)
df[ZERO_COLS] = imputer.fit_transform(df[ZERO_COLS])

print("\nAfter KNN imputation — null count:", df.isnull().sum().sum())

def cap_outliers(series, factor=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - factor * IQR, Q3 + factor * IQR)

FEATURE_COLS = [c for c in df.columns if c != "Outcome"]
for col in FEATURE_COLS:
    df[col] = cap_outliers(df[col])

print("Outlier capping applied (IQR × 3 method)")
print("\nCleaned dataset summary:")
print(df.describe().round(2))

print("\n" + "-" * 60)
print("[3] EXPLORATORY DATA ANALYSIS")
print("-" * 60)

counts = df["Outcome"].value_counts()
print(f"\nClass balance → Non-diabetic: {counts[0]}  |  Diabetic: {counts[1]}")

corr = df.corr()

fig = plt.figure(figsize=(18, 22))
fig.suptitle("Exploratory Data Analysis — Diabetes Dataset", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(["Non-diabetic (0)", "Diabetic (1)"], [counts[0], counts[1]],
               color=[BLUE, CORAL], width=0.5)
ax0.bar_label(bars, fmt="%d", padding=4)
ax0.set_title("Class distribution", fontweight="bold")
ax0.set_ylabel("Count")

ax1 = fig.add_subplot(gs[0, 1])
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=ax1, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8}
)
ax1.set_title("Feature correlation matrix", fontweight="bold")

for i, col in enumerate(FEATURE_COLS):
    row = (i // 2) + 1
    col_pos = i % 2
    ax = fig.add_subplot(gs[row, col_pos])
    for outcome, colour, label in zip([0, 1], PALETTE, ["Non-diabetic", "Diabetic"]):
        ax.hist(
            df[df["Outcome"] == outcome][col].dropna(),
            bins=25, alpha=0.65, color=colour, label=label, edgecolor="white"
        )
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    if i == 0:
        ax.legend(fontsize=9)

plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA figure saved → eda_plots.png")

fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
fig2.suptitle("Feature distributions by class (boxplot)", fontsize=13, fontweight="bold")
for ax, col in zip(axes.flatten(), FEATURE_COLS):
    df.boxplot(column=col, by="Outcome", ax=ax,
               boxprops=dict(color=BLUE),
               medianprops=dict(color=CORAL, linewidth=2),
               whiskerprops=dict(color=GRAY),
               capprops=dict(color=GRAY),
               flierprops=dict(marker="o", markerfacecolor=GRAY, markersize=3, alpha=0.4))
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("Outcome  (0 = No  |  1 = Yes)")
    ax.set_ylabel("")
fig2.suptitle("Feature distributions by class (boxplot)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Boxplot figure saved → boxplots.png")
