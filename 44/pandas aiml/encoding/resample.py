import pandas as pd

df = pd.read_csv("C:/44/pandas aiml/Transactions.csv")
print(df.info())

df['date'] = pd.to_datetime(df["date"],errors='coerce')
df['d_date'] = pd.to_datetime(df['date']).dt.strftime('%d-%m-%')
df['time'] = pd.to_datetime(df["time"])
df[['category','product_name']] = df[['category','product_name']].astype('category')
print(df.info())
print(df.head(10))
print(df.dtypes)

df['month'] = df['date'].dt.month
print(df.head(10))

# var = df['category'].unique() == "Apparel"
# print(var)
# print(pd.pivot_table(df,index="category",columns='month',values='product_name',aggfunc="count"))
print(pd.pivot_table(df,index="product_name",columns='month',values='price',aggfunc='sum'))
print(df.groupby(['category','month'])['quantity'].sum())

# pivot = pd.pivot_table(
#     df[df['category'] == 'Apparel'],
#     index='category',
#     columns='month',
#     values='product_name',
#     aggfunc='count'
# )
# print(pivot)


# Filter the DataFrame for only category "Apparel"
# df_apparel = df[df['category'] == 'Apparel']

# # Create the pivot table using this filtered data
# pivot = pd.pivot_table(
#     df_apparel,
#     index='category',
#     columns='month',
#     values='product_name',
#     aggfunc='count'
# )

# print(pivot)



# Filter only "Apparel" rows
df_apparel = df[df['category'] == 'Apparel']

# Group by 'month' and sum the 'quantity'
result = df_apparel.groupby('month')['quantity'].sum()

print(pd.pivot_table(df_apparel,index='month',values='quantity', aggfunc='sum'))
print(result)

print(df['date'].resample(('m').sum()))