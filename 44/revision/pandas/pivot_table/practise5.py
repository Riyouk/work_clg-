import pandas as pd

data = {"Gender" : ["M","F","M","F"],
        "Subjects" : ["Math","Math","Science","Science"],
        "Marks" : [80,90,85,95]}

df = pd.DataFrame(data)

pivot1 = pd.pivot_table(df,index=["Subjects"],columns="Gender",values=["Marks"],aggfunc="mean")
print(pivot1)
melt1 = pivot1.reset_index().melt(id_vars=["Subjects"],var_name="Marks",value_name="Gender")
print(melt1) 