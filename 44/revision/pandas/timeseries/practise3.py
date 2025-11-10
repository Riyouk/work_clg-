import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# -----------------------------
# 1️⃣ Create date range & values
# -----------------------------
date = pd.date_range(start="1/1/2007", end="12/31/2025", freq="D")
value = np.random.randint(10, 100, size=len(date))

df = pd.DataFrame({"Date": date, "Value": value}).set_index("Date")

# -----------------------------
# 2️⃣ Resample to monthly totals
# -----------------------------
monthly_data = df.resample("M").sum()

# Extract Year & Month for plotting
monthly_data["Year"] = monthly_data.index.year
monthly_data["Month"] = monthly_data.index.month_name()
monthly_data["Month_Num"] = monthly_data.index.month  # for sorting

# Sort properly by Year & Month
monthly_data = monthly_data.sort_values(["Year", "Month_Num"])

# -----------------------------
# 3️⃣ Create Subplots
# -----------------------------
years = monthly_data["Year"].unique()
num_years = len(years)

# Define rows/cols for subplots grid
rows = (num_years // 3) + 1   # about 3 columns per row
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
axes = axes.flatten()

# -----------------------------
# 4️⃣ Plot Each Year's Monthly Data
# -----------------------------
for i, year in enumerate(years):
    ax = axes[i]
    year_data = monthly_data[monthly_data["Year"] == year]
    
    sns.barplot(
        data=year_data,
        x="Month",
        y="Value",
        ax=ax,
        palette="viridis"
    )
    ax.set_title(f"📅 {year}", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Total Value")
    ax.tick_params(axis='x', rotation=45)

# Remove extra axes (if total subplots > num_years)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


plt.figure(figsize=(25, 7))
sns.lineplot(
    data=monthly_data,
    x=monthly_data.index,
    y="Value",
    hue="Year",
    palette="tab20",
    legend=False
)
plt.title("Monthly Trends (2007–2025)", fontsize=16, fontweight="bold")
plt.xlabel("Month-Year")
plt.ylabel("Total Value")
plt.show()
