
# import pandas as pd

# #csv file
# # read_csv = pd.read_csv('C:/44/pandas aiml/Toyota.csv',index_col=0,na_values=['??','????'])
# # cp_csv = read_csv.copy(deep=False)
# # print(cp_csv)
# # print(read_csv.head(10))

# #excel file 
# # read_xlsx = pd.read_excel('C:/44/pandas aiml/Toyata.xlsx',sheet_name='Sheet1',na_values=['??','????'],index_col=0)
# # print(read_xlsx.head(10))

# # text file 
# # read_txt = pd.read_table('C:/44/pandas aiml/Toyota.txt',index_col=0,na_values=['??','????'],sep=';',delimiter='/t')
# # print(read_txt.head(10))



# # import pandas as pd

# # Load dataset and handle missing values
# # df = pd.read_csv('C:/44/pandas aiml/Toyota.csv', index_col=0, na_values=['??', '????'])

# # # Ensure no missing values in the columns we're analyzing
# # df_clean = df[['FuelType', 'Automatic']].dropna()

# # # Group by 'Automatic' and 'FuelType'
# # grouped = df_clean.groupby(['Automatic', 'FuelType']).size().reset_index(name='Count')

# # print('🔧 Grouped Data by Transmission Type and Fuel Type:')
# # print(grouped)

# # grouping 
# df = pd.read_csv('C:/44/pandas aiml/Toyota.csv',index_col=0,na_values=['??','????'])
# # print(df.columns)
# # print(df.groupby('Sex')['Survived'].mean())

# # print(df.groupby('Automatic')['FuelType'].value_counts())
# # print(df)
# # print(df.info())
# # print(df.iloc[1])
# # print(df.loc[:3,'Price'])
# # print(df[['Price','FuelType']])
# # print(df.iloc[:10,[0,2]])
# # print(df.columns)
# # # print(df[(df['Price']>20000) & (df['FuelType']=='Petrol')].head(10))
# # print(df[df['KM']>=1000])
# # print(df[df['FuelType'].isin(['Diesel','Petrol'])])
# # print(df[df])



# #filtering missing values 
# # print(df[df['FuelType'].isna()].value_counts())
# # print(df[df['FuelType'].notna()].head())

# # df = pd.read_csv('C:/44/pandas aiml/bank[1].csv',index_col=0,sep=';')
# # print(df[df['contact']])


# #TITANIC 
# df = pd.read_csv('C:/44/pandas aiml/titanic.csv',index_col=0)
# # print(pd.pivot_table(df,index='Pclass',columns=['Sex'],values=['Survived'],aggfunc={'Survived':'sum'}))
# # print(pd.pivot_table(df,index='Pclass',columns=['Sex'],values=['Age'],aggfunc={'Age':'mean'}))
# # print(pd.pivot_table(df,index='Name',columns=['Sex'],values=['Survived'],aggfunc={'Survived':'mean'}))
# # print(pd.pivot_table(df,index='Pclass',columns=['Sex'],values=['SibSp'],aggfunc={'SibSp':['sum','mean','max','count']}))
# # print(pd.pivot_table(df,index='Embarked',columns=['Sex'],values=['Embarked'],aggfunc={'Embarked':['count']}))
# # print(pd.pivot_table(df, index='Embarked', columns='Sex', values='Age', aggfunc='count',margins=True))






# # print(df.head())
# # print(df.tail())
# # print(df.info())
# # print(df.describe())
# # print(df.columns)
# # print(df.shape)
# # print(pd.pivot_table(df,))
# # print(df[df['Survived'].max()])
# # print(df[df['Survived'] == df['Survived'].max()])
# # cabin = df[df['Cabin'].mean()]
# # print(cabin)


# # import pandas as pd

# # data = {
# #     'Department': ['IT', 'IT', 'HR', 'Finance', 'HR', 'Finance', 'IT'],
# #     'Gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M'],
# #     'Salary': [60000, 65000, 52000, 72000, 50000, 71000, 61000],
# #     'Experience': [4, 3, 2, 7, 1, 6, 5]
# # }

# # df = pd.DataFrame(data)
# # print(pd.pivot_table(df,index='Department',values=['Salary','Experience'],aggfunc={'Salary':'max','Experience':'mean'}))

# # import pandas as pd

# # # Manually entered data for 50 students
# # data = {
# #     'reg_no': [
# #         1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,
# #         1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,
# #         1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,
# #         1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,
# #         1041,1042,1043,1044,1045,1046,1047,1048,1049,1050
# #     ],
# #     'department': [
# #         'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC',
# #         'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical',
# #         'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC',
# #         'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC',
# #         'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical', 'CS', 'EC', 'Civil', 'Mechanical'
# #     ],
# #     'pms': [
# #         78, 85, 69, 72, 91, 88, 74, 60, 95, 67,
# #         83, 55, 62, 81, 70, 59, 87, 66, 77, 49,
# #         92, 84, 68, 53, 89, 76, 61, 58, 94, 80,
# #         90, 86, 65, 52, 96, 79, 73, 63, 85, 54,
# #         75, 57, 93, 82, 71, 56, 98, 64, 69, 50
# #     ],
# #     'maths': [
# #         88, 79, 84, 65, 91, 75, 82, 58, 95, 69,
# #         87, 60, 73, 80, 71, 55, 92, 66, 74, 49,
# #         94, 83, 78, 53, 85, 76, 64, 52, 97, 81,
# #         90, 86, 68, 51, 93, 79, 72, 63, 89, 54,
# #         74, 57, 96, 82, 70, 56, 99, 65, 76, 50
# #     ],
# #     'gender': [
# #         'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male',
# #         'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male',
# #         'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male',
# #         'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male',
# #         'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'
# #     ]
# # }

# # # Create DataFrame
# df = pd.DataFrame(data)

# # Display the first few rows
# # print(df[df['pms'].mean()])
# # print(df.head())
# # print(pd.pivot_table(df,index='department',values=['pms','maths'],columns='gender',aggfunc={'pms':'max','maths':'mean'}))
# print(pd.pivot_table(df,index='department',values=['pms','maths'],columns='gender',aggfunc={'pms':'sum','maths':'sum'}))

