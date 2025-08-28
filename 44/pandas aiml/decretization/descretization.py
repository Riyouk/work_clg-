import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset("titanic")

# ------------------------------
# 1. Equal-Width Binning
# ------------------------------
df["Age_bin_equal_width"] = pd.cut(df["age"], bins=4, labels=["Child", "Young", "Adult", "Senior"])

# ------------------------------
# 2. Equal-Frequency Binning (Quantile Binning)
# ------------------------------
df["Age_bin_equal_freq"] = pd.qcut(df["age"], q=4, labels=["Q1","Q2","Q3","Q4"])

# ------------------------------
# 3. Custom Binning
# ------------------------------
bins = [0, 12, 18, 35, 60, 100]  # custom age ranges
labels = ["Child", "Teen", "Young Adult", "Middle Age", "Senior"]

df["Age_bin_custom"] = pd.cut(df["age"], bins=bins, labels=labels)

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(18, 12))

# Plot Equal-Width
plt.subplot(3, 2, 1)
sns.histplot(data=df, x="age", hue="Age_bin_equal_width", multiple="stack", palette="Set2", bins=20)
plt.title("Equal-Width Binning (Histogram)")

plt.subplot(3, 2, 2)
sns.countplot(data=df, x="Age_bin_equal_width", palette="Set2")
plt.title("Equal-Width Binning (Count Plot)")

# Plot Equal-Frequency
plt.subplot(3, 2, 3)
sns.histplot(data=df, x="age", hue="Age_bin_equal_freq", multiple="stack", palette="Set3", bins=20)
plt.title("Equal-Frequency Binning (Histogram)")

plt.subplot(3, 2, 4)
sns.countplot(data=df, x="Age_bin_equal_freq", palette="Set3")
plt.title("Equal-Frequency Binning (Count Plot)")

# Plot Custom Binning
plt.subplot(3, 2, 5)
sns.histplot(data=df, x="age", hue="Age_bin_custom", multiple="stack", palette="Set1", bins=20)
plt.title("Custom Binning (Histogram)")

plt.subplot(3, 2, 6)
sns.countplot(data=df, x="Age_bin_custom", palette="Set1")
plt.title("Custom Binning (Count Plot)")

plt.tight_layout()
plt.show()
