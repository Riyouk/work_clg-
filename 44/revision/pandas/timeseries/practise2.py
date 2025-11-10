import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# data = pd.date_range(start=2024/01/01,end=2024/01/01,freq="D")
date = pd.date_range(start="1/1/2007",end="12/31/2025",freq="D")
# print(date)

value = np.random.randint(10,100,size=len(date))

df = pd.DataFrame({"Date" : date,
                    "Value" : value}).set_index("Date")
# print(df)

#slicing
# print(df.loc['2024-01-12'])
# print(df['2024-01-8': "'2024-01-15"])
# print(df.iloc[0:5])

#resampling
# print("WEEK WISE : ",df.resample('W').sum())
# print("MONTH WISE :",df.resample('M').sum())
# print("YEAR WISE : ",df.resample('Y').sum())

# print("MONTH WISE :",df.resample('M').sum() )

# june = df[df.index.year == 2007].resample('m').sum()
# print(june)

# june_sales = df[df.index.month == 4].resample("y").sum()
# june_sales.index = june_sales.index.year
# sept_sales = df[df.index.month == 9].resample("y").sum()
# sept_sales.index = june_sales.index.year


year_data = df.resample("Y").sum()
year_data.index = year_data.index.year
# print(year_data)
year_data = year_data.reset_index(names="year")

sns.barplot(data=year_data,x="year",y="Value")
plt.show()

