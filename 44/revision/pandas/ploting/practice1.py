import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/revision/pandas/EDA/sales_data.csv")
# print(df.head(10))
# print(df.describe())
# print(df.info())

# typeconversion
df[["Month","Region","Product_Category"]] = df[["Month","Region","Product_Category"]].astype("category")
print(df.info())

#1
sns.barplot(df,x="Region",y="Sales")
plt.title("HIGHEST REGION SALES TREND OVER THE YEAR")
plt.tight_layout()
plt.show()

#2 
profit_sales = df.groupby("Product_Category")[["Profit","Sales"]].sum().reset_index()
profit_sales.plot(kind="bar",x="Product_Category",y=["Profit","Sales"])
plt.title("PROFIT AND SALES TREND ACROSS PRODUCT CATEGORY")
plt.tight_layout()
plt.show()

#1.1
plt.figure(figsize=(12,6))
sns.lineplot(df,x="Month",y="Sales",hue='Region',marker="o")
plt.legend(title="REGION")
plt.title('MONTHLY SALES TREND ACROSS REGION')
plt.tight_layout()
plt.show()

# 1.2
# sns.barplot(data=df,x="Month",y="Sales",hue="Product_Category")
# plt.title("PRODUCT CATEGORY SALES TREND ACROSS MONTH")
# plt.tight_layout()
# plt.show()
#1.2
data = df.groupby(["Month", "Product_Category"])["Sales"].sum().unstack()
data.plot(kind="bar", stacked=False)
plt.title("Product Category Contribution to Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

#1.3
sns.barplot(data=df, x="Region", y="Sales", estimator=sum)
plt.title("Total Annual Sales per Region")
plt.tight_layout()
plt.show()

#1.4
# region = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index()
# fig, ax1 = plt.subplots()
# sns.barplot(data=region, x="Region", y="Sales", ax=ax1, color="lightblue")
# ax2 = ax1.twinx()
# sns.lineplot(data=region, x="Region", y="Profit", ax=ax1, color="red", marker="o")
# plt.title("Sales and Profit per Region")
# plt.show()

#1.4
profit_sales1 = df.groupby("Region")[["Profit","Sales"]].sum().reset_index()
profit_sales1.plot(kind="bar",x="Region",y=["Profit","Sales"])
plt.title("PROFIT AND SALES TREND ACROSS REGION")
plt.tight_layout()
plt.show()

#2.1
sns.scatterplot(data=df, x='Sales', y='Profit', hue='Region', s=100, palette='coolwarm')
plt.title("Sales vs Profit by Region")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend(title='Region')
plt.show()

#2.2
df1 = df.copy()
df1["Margin"] = (df["Profit"].div(df["Sales"]))*100
# print(df1)
sns.boxplot(data=df1,x="Product_Category",y="Margin",palette="Set2")
plt.show()
sns.violinplot(data=df1,x="Product_Category",y="Margin",palette="Set2")
plt.show()

# 2.2
df1["pro_sal"] = df1["Profit"]/df1["Sales"] * 100
sns.boxplot(data=df1,x="Product_Category",y="pro_sal",palette="Set2")
plt.show()

# 2.3
top_three = df1.groupby("Region")["Profit"].agg("sum").head(3).reset_index().sort_values(by="Profit",ascending=False)
print(top_three)
sns.barplot(data=top_three,x="Profit",y="Region",palette="Set2")
plt.show()

# 2.4
print(df["Product_Category"].unique())
encode = {'Electronics': 0, 'Furniture': 1, 'Groceries': 2, 'Clothing': 3}
df["Product_Category"] = df["Product_Category"].map(encode)
print(df)
num_col = df.select_dtypes(include=(np.number))
print(num_col)
corr = num_col.corr()
print(corr)
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()

# 3.2
df["Month"] = pd.Categorical(df["Month"], categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered=True)
g = sns.FacetGrid(df1, col="Region", height=4, aspect=1.5)
g.map_dataframe(sns.lineplot, x="Month", y="Sales",marker="o")
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45)
plt.show()

# 3.3
sns.lineplot(data=df1, x="Month", y="Profit", hue="Region", marker="o",estimator="mean")
plt.show()

# 4.1
sns.barplot(data=df1,x="Product_Category",y="Sales",hue="Region",estimator="sum")
plt.show()

# 4.2
profit_by_category = df1.groupby("Product_Category")["Profit"].sum()
plt.pie(profit_by_category, labels=profit_by_category.index, autopct="%1.1f%%")
plt.title("Share of Total Profit by Product Category")
plt.show()

# 4.3
sns.barplot(data=df1,x="Product_Category",y="Sales",estimator="sum")
sns.lineplot(data=df1,x="Product_Category",y="Profit",estimator="mean",marker="o",color="red")
plt.show()

# 4.4
# # Plotly Treemap for hierarchical data visualization
# import plotly.express as px
# import plotly.graph_objects as go

# # Create hierarchical data structure for treemap
# # Group by Region and Product Category to get sales data
# treemap_data = df.groupby(['Region', 'Product_Category'])['Sales'].sum().reset_index()
# treemap_data['Profit'] = df.groupby(['Region', 'Product_Category'])['Profit'].sum().reset_index()['Profit']

# # Create treemap with Plotly
# fig = px.treemap(
#     treemap_data,
#     path=['Region', 'Product_Category'],  # Hierarchy: Region -> Product Category
#     values='Sales',
#     color='Sales',
#     hover_data=['Profit'],
#     color_continuous_scale='Viridis',
#     title='Hierarchical Sales Data: Region → Product Category'
# )

# fig.update_layout(
#     width=1000,
#     height=600,
#     font=dict(size=12)
# )

# fig.show()

# # Alternative treemap with custom colors and better formatting
# fig2 = go.Figure(go.Treemap(
#     labels=treemap_data['Product_Category'],
#     parents=treemap_data['Region'],
#     values=treemap_data['Sales'],
#     text=treemap_data.apply(lambda x: f"Sales: ${x['Sales']:,.0f}<br>Profit: ${x['Profit']:,.0f}", axis=1),
#     hovertemplate='<b>%{parent} → %{label}</b><br>%{text}<extra></extra>',
#     marker=dict(
#         colors=treemap_data['Sales'],
#         colorscale='RdYlBu',
#         showscale=True,
#         colorbar=dict(title="Sales ($)")
#     ),
#     textfont=dict(size=14, color='white'),
#     pathbar=dict(visible=True)
# ))

# fig2.update_layout(
#     title='Sales Distribution by Region and Product Category',
#     width=1000,
#     height=600,
#     font=dict(size=12)
# )

# fig2.show()

# # Create a more detailed treemap with three levels (if you have subcategories)
# # This example shows how to add a third level if needed
# detailed_data = df.groupby(['Region', 'Product_Category', 'Month'])['Sales'].sum().reset_index()
# detailed_data['Profit'] = df.groupby(['Region', 'Product_Category', 'Month'])['Profit'].sum().reset_index()['Profit']

# fig3 = px.treemap(
#     detailed_data,
#     path=['Region', 'Product_Category', 'Month'],  # Three-level hierarchy
#     values='Sales',
#     color='Profit',
#     hover_data=['Sales'],
#     color_continuous_scale='RdYlGn',
#     title='Three-Level Hierarchy: Region → Product Category → Month'
# )

# fig3.update_layout(
#     width=1200,
#     height=700,
#     font=dict(size=10)
# )

# fig3.show()

