
import pandas as pd

#csv file
# read_csv = pd.read_csv('C:/44/pandas aiml/Toyota.csv',index_col=0,na_values=['??','????'])
# cp_csv = read_csv.copy(deep=False)
# print(cp_csv)
# print(read_csv.head(10))

#excel file 
# read_xlsx = pd.read_excel('C:/44/pandas aiml/Toyata.xlsx',sheet_name="Sheet1",na_values=['??','????'],index_col=0)
# print(read_xlsx.head(10))

# text file 
# read_txt = pd.read_table('C:/44/pandas aiml/Toyota.txt',index_col=0,na_values=['??','????'],sep=";",delimiter="\t")
# print(read_txt.head(10))



# import pandas as pd

# Load dataset and handle missing values
# df = pd.read_csv('C:/44/pandas aiml/Toyota.csv', index_col=0, na_values=['??', '????'])

# # Ensure no missing values in the columns we're analyzing
# df_clean = df[['FuelType', 'Automatic']].dropna()

# # Group by 'Automatic' and 'FuelType'
# grouped = df_clean.groupby(['Automatic', 'FuelType']).size().reset_index(name='Count')

# print("🔧 Grouped Data by Transmission Type and Fuel Type:")
# print(grouped)

# # grouping 
df = pd.read_csv('C:/44/pandas aiml/Toyota.csv',index_col=0,na_values=['??','????'])
# # print(df.columns)

# # # print(df.groupby('Automatic')['FuelType'].value_counts())
# # # print(df)
# # # print(df.info())
print(df.iloc[1])
