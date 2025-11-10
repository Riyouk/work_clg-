# import pandas as pd

# # Depot 1 data
# D1 = pd.DataFrame({
#     'Vehicle': ['V1', 'V2', 'V3'],
#     'Mon': [12, 15, 13],
#     'Tue': [14, 16, 14],
#     'Wed': [13, 14, 15],
#     'Thu': [12, 15, 13],
#     'Fri': [11, 15, 14]
# })

# # Depot 2 data
# D2 = pd.DataFrame({
#     'Vehicle': ['V1', 'V2', 'V3'],
#     'Mon': [11, 14, 15],
#     'Tue': [12, 16, 14],
#     'Wed': [13, 14, 15],
#     'Thu': [11, 15, 14],
#     'Fri': [12, 15, 13]
# })

# # Depot 3 data
# D3 = pd.DataFrame({
#     'Vehicle': ['V1', 'V2', 'V3'],
#     'Mon': [13, 16, 14],
#     'Tue': [14, 17, 15],
#     'Wed': [15, 14, 15],
#     'Thu': [14, 16, 13],
#     'Fri': [13, 15, 14]
# })


# combined_df = pd.concat([D1,D2,D3],ignore_index=True)
# print(combined_df)

# print(pd.pivot_table(combined_df,index=["Vehicle"],values=["Mon","Tue","Wed","Thu","Fri"],aggfunc="mean"))
# print(pd.pivot_table(combined_df,index=["Vehicle"],values=["Mon","Tue","Wed","Thu","Fri"],aggfunc="std"))







# ============================================================
# 🚚 Logistics Company Fuel Consumption Analysis
# ============================================================
# Task:
# 1. Combine all depot data (D1, D2, D3) into one consolidated DataFrame.
# 2. Compute the weekly average fuel consumption per vehicle.
# 3. Identify the vehicle with the most consistent consumption trend.
# ============================================================

import pandas as pd

# ----------------------------
# STEP 1: Create sample data
# ----------------------------

# Depot 1 data (Fuel usage in liters)
D1 = pd.DataFrame({
    'Vehicle': ['V1', 'V2', 'V3'],
    'Mon': [12, 15, 13],
    'Tue': [14, 16, 14],
    'Wed': [13, 14, 15],
    'Thu': [12, 15, 13],
    'Fri': [11, 15, 14]
})

# Depot 2 data
D2 = pd.DataFrame({
    'Vehicle': ['V1', 'V2', 'V3'],
    'Mon': [11, 14, 15],
    'Tue': [12, 16, 14],
    'Wed': [13, 14, 15],
    'Thu': [11, 15, 14],
    'Fri': [12, 15, 13]
})

# Depot 3 data
D3 = pd.DataFrame({
    'Vehicle': ['V1', 'V2', 'V3'],
    'Mon': [13, 16, 14],
    'Tue': [14, 17, 15],
    'Wed': [15, 14, 15],
    'Thu': [14, 16, 13],
    'Fri': [13, 15, 14]
})

# ----------------------------
# STEP 2: Combine all depots
# ----------------------------

combined_df = pd.concat([D1, D2, D3], ignore_index=True)
print("=== Combined DataFrame ===")
print(combined_df)
print("\n")

# ----------------------------
# STEP 3: Compute Weekly Average Fuel Consumption
# ----------------------------

# Calculate average fuel per week (for each depot entry)
combined_df['Weekly_Avg'] = combined_df[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean(axis=1)
print(combined_df)

# Compute mean weekly average for each vehicle across depots
avg_consumption = combined_df.groupby('Vehicle')['Weekly_Avg'].mean().reset_index()
avg_consumption.columns = ['Vehicle', 'Avg_Fuel_Consumption']

print("=== Weekly Average Fuel Consumption per Vehicle ===")
print(avg_consumption)
print("\n")

# ----------------------------
# STEP 4: Identify Most Consistent Vehicle
# ----------------------------

# Melt the dataset to long format (day-wise)
melted_df = combined_df.melt(id_vars='Vehicle', var_name='Day', value_name='Fuel')

# Compute standard deviation (variation in fuel usage)
std_df = melted_df.groupby('Vehicle')['Fuel'].std().reset_index()
std_df.columns = ['Vehicle', 'Fuel_Std_Deviation']

# Identify vehicle with lowest std deviation (most consistent)
most_consistent = std_df.loc[std_df['Fuel_Std_Deviation'].idxmin()]

print("=== Standard Deviation (Fuel Variation) per Vehicle ===")
print(std_df)
print("\n")

print("=== Vehicle with Most Consistent Fuel Consumption ===")
print(most_consistent)
print("\n")

# ============================================================
# ✅ Final Summary
# ============================================================
print("Summary:")
print(f"- Combined depots: 3 (D1, D2, D3)")
print(f"- Total records combined: {len(combined_df)}")
print(f"- Vehicle with most consistent trend: {most_consistent['Vehicle']}")
print("============================================================")