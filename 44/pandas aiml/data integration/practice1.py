import pandas as pd 

orders_file = pd.read_csv("C:/44/pandas aiml/data integration/orders.csv")
products_file = pd.read_csv("C:/44/pandas aiml/data integration/products.csv")

# comined = pd.merge(orders_file,products_file,how="right",)
# comined1= pd.merge(orders_file,products_file,how="left",)
# comined = pd.merge(orders_file,products_file,how="inner",on='product_id')
# comined['total_price'] = comined['quantity']*comined['price']
merge_hr = pd.concat([orders_file,products_file],axis=1)
# print(comined)
print(merge_hr.head(50))

