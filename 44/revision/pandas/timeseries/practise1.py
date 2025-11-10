import numpy as np 
import pandas as pd

# data = pd.date_range(start=1/1/2024,end=1/10/2024,freq="D")
date = pd.date_range(start="1/1/2024",end="1/10/2024",freq="D")
# print(date)

value = np.random.randint(10,100,size=len(date))

df = pd.DataFrame({"Date" : date,
                    "Value" : value}).set_index("Date")
print(df)
print(df.loc['2024-01-12'])
print(df['2024-01-8': "'2024-01-15"])
print(df.iloc[1:3])


