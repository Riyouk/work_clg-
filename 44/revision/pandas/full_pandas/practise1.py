# ============================================================
# 🧠 PANDAS ALL-IN-ONE MASTER SCRIPT
# Author: ChatGPT (GPT-5)
# Description: Complete demonstration of all major pandas functions
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1️⃣ SERIES CREATION AND BASIC OPERATIONS
# ------------------------------------------------------------

# Example 1: Create a Series from list
s1 = pd.Series([10, 20, 30, 40])
print("Series 1:\n", s1, "\n")

# Example 2: Create a Series with custom index
s2 = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
print("Series 2 with custom index:\n", s2, "\n")

# Example 3: Create Series from dictionary
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print("Series from dictionary:\n", s3, "\n")

# Accessing elements
print("Access by index:", s3['b'])
print("Access multiple:", s3[['a', 'c']], "\n")

# ------------------------------------------------------------
# 2️⃣ DATAFRAME CREATION AND EXPLORATION
# ------------------------------------------------------------

data = {
    'Name': ['Amit', 'Ravi', 'Sneha', 'Meena'],
    'Age': [25, 30, 22, 28],
    'City': ['Delhi', 'Mumbai', 'Chennai', 'Pune']
}

df = pd.DataFrame(data)
print("Initial DataFrame:\n", df, "\n")

# Example 1: Basic info
print("DataFrame Info:")
print(df.info(), "\n")

# Example 2: Describe numeric columns
print("Descriptive Statistics:\n", df.describe(), "\n")

# Example 3: Shape, Columns, Dtypes
print("Shape:", df.shape)
print("Columns:", df.columns)
print("Data types:\n", df.dtypes, "\n")

# ------------------------------------------------------------
# 3️⃣ DATA SELECTION AND INDEXING
# ------------------------------------------------------------

# Example 1: Select single column
print("Select column Name:\n", df['Name'], "\n")

# Example 2: Select multiple columns
print("Select Name and City:\n", df[['Name', 'City']], "\n")

# Example 3: Select row by label and position
print("Row 2 (iloc):\n", df.iloc[2], "\n")
print("Row with index 1 (loc):\n", df.loc[1], "\n")

# Example 4: Specific cell
print("Cell value (Ravi's City):", df.loc[1, 'City'], "\n")

# ------------------------------------------------------------
# 4️⃣ FILTERING DATA
# ------------------------------------------------------------

# Example 1: Filter by single condition
print("Age > 25:\n", df[df['Age'] > 25], "\n")

# Example 2: Multiple conditions
print("Age > 25 and City == 'Pune':\n", df[(df['Age'] > 25) & (df['City'] == 'Pune')], "\n")

# Example 3: Using isin()
print("Filter where City in ['Delhi', 'Pune']:\n", df[df['City'].isin(['Delhi', 'Pune'])], "\n")

# ------------------------------------------------------------
# 5️⃣ ADDING & REMOVING DATA
# ------------------------------------------------------------

# Add new column
df['Salary'] = [50000, 60000, 45000, 52000]
print("After adding Salary:\n", df, "\n")

# Add derived column
df['Bonus'] = df['Salary'] * 0.10
print("After adding Bonus:\n", df, "\n")

# Remove column
df.drop('Bonus', axis=1, inplace=True)
print("After removing Bonus:\n", df, "\n")

# Remove row
df.drop(2, inplace=True)
print("After dropping row 2:\n", df, "\n")

# ------------------------------------------------------------
# 6️⃣ MISSING DATA HANDLING
# ------------------------------------------------------------

df2 = pd.DataFrame({
    'A': [1, np.nan, 3, 4],
    'B': [np.nan, 2, 3, np.nan]
})
print("DataFrame with Missing Values:\n", df2, "\n")

# Example 1: Drop missing rows
print("Drop NA:\n", df2.dropna(), "\n")

# Example 2: Fill missing values
print("Fill with zeros:\n", df2.fillna(0), "\n")

# Example 3: Fill with column mean
print("Fill with mean:\n", df2.fillna(df2.mean(numeric_only=True)), "\n")

# ------------------------------------------------------------
# 7️⃣ SORTING AND RANKING
# ------------------------------------------------------------

print("Sort by Age:\n", df.sort_values(by='Age', ascending=False), "\n")
print("Sort by index:\n", df.sort_index(), "\n")

# ------------------------------------------------------------
# 8️⃣ GROUPBY AND AGGREGATION
# ------------------------------------------------------------

df3 = pd.DataFrame({
    'City': ['Delhi', 'Delhi', 'Mumbai', 'Pune', 'Pune'],
    'Sales': [200, 150, 300, 250, 100],
    'Profit': [50, 40, 70, 60, 30]
})

print("Group by City (Mean Sales):\n", df3.groupby('City')['Sales'].mean(), "\n")
print("Group by City (Multiple Agg):\n", df3.groupby('City').agg({'Sales': 'sum', 'Profit': 'mean'}), "\n")

# ------------------------------------------------------------
# 9️⃣ MERGE, JOIN, CONCAT
# ------------------------------------------------------------

dfA = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
dfB = pd.DataFrame({'ID': [2, 3, 4], 'Score': [80, 90, 70]})

# Merge
print("Merge (inner):\n", pd.merge(dfA, dfB, on='ID', how='inner'), "\n")

# Join (based on index)
dfA.set_index('ID', inplace=True)
dfB.set_index('ID', inplace=True)
print("Join (outer):\n", dfA.join(dfB, how='outer'), "\n")

# Concatenate
dfC = pd.DataFrame({'X': [1, 2], 'Y': [3, 4]})
dfD = pd.DataFrame({'X': [5, 6], 'Y': [7, 8]})
print("Concatenate vertically:\n", pd.concat([dfC, dfD]), "\n")

# ------------------------------------------------------------
# 🔟 APPLY, MAP, LAMBDA, FILTER
# ------------------------------------------------------------

df['Updated_Age'] = df['Age'].apply(lambda x: x + 2)
print("Apply Example:\n", df, "\n")

df['City_Code'] = df['City'].map({'Delhi': 'DL', 'Mumbai': 'MB', 'Pune': 'PN'})
print("Map Example:\n", df, "\n")

# Filter using apply
print("Filter Salary > 52000:\n", df[df['Salary'].apply(lambda x: x > 52000)], "\n")

# ------------------------------------------------------------
# 11️⃣ MELT & PIVOT (RESHAPING)
# ------------------------------------------------------------

df_reshape = pd.DataFrame({
    'Name': ['A', 'B', 'C'],
    'Math': [80, 70, 90],
    'Science': [85, 75, 95]
})

# Melt: wide → long
melted = pd.melt(df_reshape, id_vars=['Name'], value_vars=['Math', 'Science'], var_name='Subject', value_name='Score')
print("Melted DataFrame:\n", melted, "\n")

# Pivot: long → wide
pivoted = melted.pivot(index='Name', columns='Subject', values='Score')
print("Pivoted DataFrame:\n", pivoted, "\n")

# ------------------------------------------------------------
# 12️⃣ STATISTICS AND ANALYSIS
# ------------------------------------------------------------

print("Mean Age:", df['Age'].mean())
print("Correlation Matrix:\n", df.corr(numeric_only=True), "\n")
print("Value Counts for City:\n", df['City'].value_counts(), "\n")

# ------------------------------------------------------------
# 13️⃣ STRING OPERATIONS
# ------------------------------------------------------------

df['Name'] = df['Name'].str.upper()
print("Uppercase Names:\n", df, "\n")

df['City'] = df['City'].str.replace('i', '*', regex=False)
print("City Replace Example:\n", df, "\n")

# ------------------------------------------------------------
# 14️⃣ DATETIME OPERATIONS
# ------------------------------------------------------------

date_df = pd.DataFrame({
    'Date': pd.to_datetime(['2025-01-01', '2025-03-15', '2025-07-20'])
})
date_df['Year'] = date_df['Date'].dt.year
date_df['Month'] = date_df['Date'].dt.month
date_df['Day'] = date_df['Date'].dt.day
print("DateTime Example:\n", date_df, "\n")

# ------------------------------------------------------------
# 15️⃣ EXPORTING DATA
# ------------------------------------------------------------

# Save to CSV / Excel (commented to prevent file creation during demo)
# df.to_csv('output.csv', index=False)
# df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
print("✅ Data export functions demonstrated (to_csv / to_excel).")

# ============================================================
# ✅ END OF PANDAS ALL-IN-ONE MASTER SCRIPT
# ============================================================
