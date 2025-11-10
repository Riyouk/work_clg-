import pandas as pd 
data = {"month": ["jan","jan" ,"feb","feb"],
        "product" : ["a","b","a","b"],
        "sales" : [100,200,150,250],
        "profit" : [30,70,50,100]}
df = pd.DataFrame(data)
print(df.columns)


# print(pd.pivot_table(df,index=["month"],columns=["product"],values=["sales"],aggfunc="sum"))
# print(pd.pivot_table(df,index=["product"],columns=["month"],values=["sales"],aggfunc="sum"))
# print(pd.pivot_table(df,index=["product"],columns=["month"],values=["sales","profit"],aggfunc="sum",sort=False))


pivot1 =pd.pivot_table(df,index=["month"],columns=["product"],values=["sales"],aggfunc="sum")
melt1 = pivot1.reset_index().melt(id_vars=["month"],var_name="product",value_name="Sales")
print("melt 1 : \n ",melt1)

pivot2 =pd.pivot_table(df,index=["product"],columns=["month"],values=["sales"],aggfunc="sum")
melt2 = pivot2.reset_index().melt(id_vars=["product"],var_name="Month",value_name="Sales")
print("melt 2 : \n ",melt2)

pivot3 = pd.pivot_table(df,index=["product"],columns=["month"],values=["sales","profit"],aggfunc="sum",sort=False)





# def pivot(data,index,columns,values,aggfunc):
#     print(pd.pivot_table(data,index=[index],columns=[columns],values=[values],aggfunc={aggfunc}))

# pivot(df,"month","product","sales","sum")