# import pandas as pd 

# df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/revision/pandas/pivot_table/student_scores.csv")
# print(df)

# # print(pd.pivot_table(df,index=["GENDER"],values=["CIE","DA","ASSNT"],aggfunc={max}))
# # print(pd.pivot_table(df,index=["GENDER"],values=["CIE","DA","ASSNT"],aggfunc={"mean"}))
# # print(pd.pivot_table(df,index=["GENDER"],values=["CIE","DA","ASSNT"],aggfunc={"min"}))
# # print(pd.pivot_table(df,columns=["GENDER"],values=["CIE","DA","ASSNT"],aggfunc={"std"}))

# # df["total"] = df[["CIE","DA","ASSNT"]].agg(func=sum)
# df["total"] = df["CIE"]+df["DA"]+df["ASSNT"]

# print(df["total"])

# df["DA_per"] = df["DA"]*100/20
# df["ASSNT_per"] = df["ASSNT"]*100/150
# df["CIE_per"] = df["ASSNT"]*100/30















# =============================================================
# 📊 PIVOT TABLES USING BINS AND LABELS
# =============================================================

import pandas as pd

# -------------------------------------------------------------
# 1️⃣ CREATE DATAFRAME
# -------------------------------------------------------------
data = {
    'GENDER': ['B','G','G','G','B','B','B','G','B','B','G','B','G','B','B','G','G','B','B','B','G','B','B','B','G','B','G'],
    'DA': [9,13,6,11,12,14,18,19,12,11,12,15,9,13,13,10,8,4,7,12,17,14,18,10,11,14,12],
    'CIE': [45,134,125,120,72,107,146,150,36,129,147,138,82,110,36,107,103,24,42,42,129,99,133,69,94,72,70],
    'ASSNT': [8,10,10,9,8,9,10,10,7,10,10,10,7,10,7,9,8,8,7,8,10,9,10,7,8,8,8]
}

df = pd.DataFrame(data)

print("ORIGINAL DATA (first 5 rows):")
print(df.head(), "\n")

# -------------------------------------------------------------
# 2️⃣ CREATE BINS AND LABELS FOR CIE (Continuous Marks)
# -------------------------------------------------------------
cie_bins = [0, 50, 100, 150]
cie_labels = ['Low', 'Medium', 'High']

# Create a new categorical column based on bins
df['CIE_Level'] = pd.cut(df['CIE'], bins=cie_bins, labels=cie_labels)

# Verify grouping
print("Data with CIE Levels:\n", df[['CIE', 'CIE_Level']].head(10), "\n")

# -------------------------------------------------------------
# 3️⃣ PIVOT TABLES USING BINS
# -------------------------------------------------------------

# 3.1️⃣ Mean DA and ASSNT grouped by GENDER and CIE Level
pivot1 = pd.pivot_table(df, index=['GENDER', 'CIE_Level'], values=['DA', 'ASSNT'], aggfunc='mean')
print("1️⃣ MEAN DA & ASSNT BY GENDER AND CIE LEVEL:\n", pivot1, "\n")

# -------------------------------------------------------------
# 3.2️⃣ Count of Students in Each CIE Level by Gender
pivot2 = pd.pivot_table(df, index='CIE_Level', columns='GENDER', values='CIE', aggfunc='count', fill_value=0)
print("2️⃣ COUNT OF STUDENTS IN EACH CIE LEVEL BY GENDER:\n", pivot2, "\n")

# -------------------------------------------------------------
# 3.3️⃣ Average CIE per DA Range
da_bins = [0, 10, 15, 20]
da_labels = ['Low', 'Mid', 'High']
df['DA_Level'] = pd.cut(df['DA'], bins=da_bins, labels=da_labels)

pivot3 = pd.pivot_table(df, index='DA_Level', values='CIE', aggfunc=['mean', 'max', 'min'])
print("3️⃣ CIE STATS BY DA LEVEL:\n", pivot3, "\n")

# -------------------------------------------------------------
# 3.4️⃣ Two-level Pivot: DA Level × CIE Level
pivot4 = pd.pivot_table(df, index='DA_Level', columns='CIE_Level', values='ASSNT', aggfunc='mean')
print("4️⃣ MEAN ASSNT SCORE BY DA LEVEL AND CIE LEVEL:\n", pivot4, "\n")

# -------------------------------------------------------------
# 3.5️⃣ Adding Margins (Totals)
pivot5 = pd.pivot_table(df, index='CIE_Level', values='CIE', aggfunc='mean', margins=True)
print("5️⃣ MEAN CIE BY LEVEL (WITH TOTAL):\n", pivot5, "\n")

# -------------------------------------------------------------
# 3.6️⃣ Visual-Ready Pivot (Style)
pivot6 = pd.pivot_table(df, index='CIE_Level', columns='GENDER', values='DA', aggfunc='mean')
styled_pivot = pivot6.style.background_gradient(cmap='coolwarm')
print("6️⃣ STYLED PIVOT (Use in Jupyter Notebook for color view):\n", pivot6, "\n")

# =============================================================
# OPTIONAL EXPORT TO EXCEL
# =============================================================
# with pd.ExcelWriter("Pivot_with_Bins.xlsx") as writer:
#     pivot1.to_excel(writer, sheet_name="Mean_DA_Assnt_by_Level")
#     pivot2.to_excel(writer, sheet_name="Count_by_CIE_Level")
#     pivot3.to_excel(writer, sheet_name="CIE_by_DA_Level")
#     pivot4.to_excel(writer, sheet_name="Cross_DA_CIE")
#     pivot5.to_excel(writer, sheet_name="Mean_CIE_with_Total")
# print("✅ All pivot tables exported to Pivot_with_Bins.xlsx")
