# import pandas as pd 
# data = {"region" : ["north","south"],
#         "product" : ["phone","laptop"],
#         "jan" : [23000,18000],
#         "feb" : [18000,21000],
#         "mar" : [21000,23000],
#         "apr" : [30000,25000]}
# df = pd.DataFrame(data)
# # print(df.columns)

# # print(df.reset_index().melt(id_vars=["product"],value_vars=['jan','feb','mar','apr'],var_name="products",value_name="month-wise"))

# df["monthly_avg"] = df[['jan','feb','mar','apr']].mean(axis=1)
# # print(df)

# # avg_monthly_sales = df.groupby("product")['monthly_avg'].mean().reset_index()
# print(pd.pivot_table(df,index=["product"],columns=["region"],values=['jan','feb','mar','apr'],aggfunc="mean",fill_value=0,margins=True,margins_name="TOTAL"))
# # print(avg_monthly_sales)

# print(df.reset_index().melt())




# ============================================================
# 🏪 Retail Chain Monthly Sales Analysis
# ============================================================
# Tasks:
# 1️⃣ Transform the dataset to a long format (month-wise sales records)
# 2️⃣ Compute the average monthly sales per product across regions
# 3️⃣ Identify the month with the highest average growth
# ============================================================

# Step 1: Import the required library
import pandas as pd

# ------------------------------------------------------------
# Step 2: Create the DataFrame (given dataset)
# ------------------------------------------------------------
data = {
    'Region': ['North', 'South'],
    'Product': ['Phone', 'Laptop'],
    'Jan': [23000, 18000],
    'Feb': [25000, 21000],
    'Mar': [27000, 23000],
    'Apr': [30000, 25000]
}

df = pd.DataFrame(data)
print("=== Original DataFrame ===")
print(df)
print("\n")

# ------------------------------------------------------------
# Step 3: Transform the dataset to long format
# ------------------------------------------------------------
# pd.melt() converts columns into rows — useful for time-series data
long_df = pd.melt(
    df,
    id_vars=['Region', 'Product'],  # Keep Region & Product fixed
    var_name='Month',               # New column for months
    value_name='Sales'              # New column for sales values
)

print("=== Long Format DataFrame ===")
print(long_df)
print("\n")

# ------------------------------------------------------------
# Step 4: Compute the average monthly sales per product
# ------------------------------------------------------------
# Group data by Product and Month, then calculate mean of sales
avg_sales = long_df.groupby(['Product', 'Month'])['Sales'].mean().reset_index()

print("=== Average Monthly Sales per Product ===")
print(avg_sales)
print("\n")

# ------------------------------------------------------------
# Step 5: Compute overall average monthly sales (across all regions)
# ------------------------------------------------------------
# Group data by Month only
monthly_avg = long_df.groupby('Month')['Sales'].mean().reset_index()

print("=== Average Monthly Sales Across Regions ===")
print(monthly_avg)
print("\n")

# ------------------------------------------------------------
# Step 6: Calculate month-to-month growth and identify highest growth month
# ------------------------------------------------------------

# Define correct month order
month_order = ['Jan', 'Feb', 'Mar', 'Apr']
monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
monthly_avg = monthly_avg.sort_values('Month')

# Compute month-to-month growth (difference from previous month)
monthly_avg['Growth'] = monthly_avg['Sales'].diff()

# Identify the month with the highest average growth
max_growth_month = monthly_avg.loc[monthly_avg['Growth'].idxmax()]

print("=== Monthly Growth Data ===")
print(monthly_avg)
print("\n")

print("=== Month with Highest Average Growth ===")
print(max_growth_month)
print("\n")

# ------------------------------------------------------------
# Step 7: Final Summary
# ------------------------------------------------------------
print("============================================================")
print("📊 FINAL SUMMARY")
print("------------------------------------------------------------")
print("1️⃣ Long format transformation completed using pd.melt()")
print("2️⃣ Average monthly sales per product computed using groupby() and mean()")
print("3️⃣ Month with highest average growth identified using diff() and idxmax()")
print("------------------------------------------------------------")
print(f"➡ Highest average growth month: {max_growth_month['Month']} (+{max_growth_month['Growth']} sales increase)")
print("============================================================")