# import pandas as pd 
from matplotlib import pyplot as plt 
# df = pd.read_csv('C:/44/pandas aiml/titanic.csv',index_col=0)
# # print(df.head())
# fig = plt.figure()

import pandas as pd

# data = {
#     'Department': ['IT', 'IT', 'HR', 'Finance', 'HR', 'Finance', 'IT'],
#     'Gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M'],
#     'Salary': [60000, 65000, 52000, 72000, 50000, 71000, 61000],
#     'Experience': [4, 3, 2, 7, 1, 6, 5]
# }

# df = pd.DataFrame(data)
# # plt.scatter(df['Department'],df['Salary'])
# plt.scatter(df['Experience'],df['Salary'])
# plt.show()

#scatter plot 
# read_csv = pd.read_csv('C:/44/pandas aiml/Toyota.csv',index_col=0,na_values=['??','????'])
# plt.scatter(read_csv['Age'],read_csv['Price'],c='purple',s=100,alpha=0.5,marker='D',edgecolors="green")
# # plt.scatter(read_csv[''],read_csv['HP'],c='purple')
# plt.title("how CC of engine increases the Horse power of the car ")
# plt.legend("price")
# # plt.subplot(1,2,1)
# # plt.grid()
# plt.xlim(0,80)
# plt.ylim(0,30000)
# plt.xticks(range(0,80,10))
# plt.yticks(range(0,30000,1000))
# plt.xlabel('CC')
# plt.ylabel('HP')
# # plt.grid(True,linestyle='--',linewidth='2.0')
# plt.show()


#histogram 
# read_csv = pd.read_csv('C:/44/pandas aiml/Toyota.csv',index_col=0,na_values=['??','????'])
# histogram = read_csv.dropna()
# # print(read_csv)
# fuel_types = histogram['FuelType'].unique()
# plt.title("DATA ON FUEL TYPE ⛽")
# plt.hist(histogram['FuelType'],bins=10,color='red')
# plt.xlabel("fuel types")
# plt.ylabel("Count of cars ")

# fuel_types = histogram['FuelType'].unique()

# for fuel in fuel_types:
#     plt.hist(histogram[histogram['FuelType'] == fuel]['FuelType'],
#              bins=1, label=fuel, alpha=0.7)

# plt.legend(title="Fuel Types")
# # plt.legend(fuel_types,)
# plt.show()







# data = {
#     'Year': [2019, 2020, 2021, 2022, 2023],
#     'Gold_Price_USD': [1350, 1550, 1800, 1750, 1900],      # Avg price per ounce
#     'Silver_Price_USD': [1200, 1400, 1600, 1850, 10000],    # Avg price per ounce
#     'Gold_Change_%': [2.1, 14.8, 16.1, -2.8, 8.6],
#     'Silver_Change_%': [3.5, 22.3, 36.5, -10.2, 4.8]
# }

# import pandas as pd
# df = pd.DataFrame(data)
# print(df)
# plt.scatter(df['Year'],df['Gold_Price_USD'],c='red',marker='D',label="GOLD")
# plt.scatter(df['Year'],df['Silver_Price_USD'],c='green',marker='D',label="SILVER")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.xlim(2019,2023)
# plt.ylim(1350,2000)
# plt.xticks(range(2019,2025),rotation=60)
# plt.yticks(range(1350,2000,200))
# plt.legend(loc="best",bbox_to_anchor=(1,1))
# plt.grid()
# plt.show()

#figure
# # x = [10,20,30,40,50,60,70,80]
# # plt.plot(x)
# fig = plt.figure(figsize=(10,8),dpi=100)
# fig.suptitle("My First plot")
# ax1 = fig.add_subplot(3,2,1)
# ax2 = fig.add_subplot(3,2,2)
# # fig1,axs = plt.subplots(1,2, figsize=(10,4),dpi=100)

# plt.show()


#method 1 
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))

# # First subplot (1 row, 2 columns, 1st plot)
# plt.subplot(1, 2, 1)
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.title("Left Plot")

# # Second subplot
# plt.subplot(1, 2, 2)
# plt.plot([1, 2, 3], [6, 5, 4])
# plt.title("Right Plot")

# plt.tight_layout()  # Avoid overlap
# plt.show()

#method 2
# import matplotlib.pyplot as plt

# # Create 2 subplots (1 row, 2 columns)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # Plotting on the first axis
# axes[0].plot([1, 2, 3], [3, 2, 1])
# axes[0].set_title("Plot 1")
# axes[0].xlabel()


# # Plotting on the second axis
# axes[1].plot([1, 2, 3], [1, 2, 3])
# axes[1].set_title("Plot 2")

# plt.tight_layout()
# plt.show()


#bar graph 
# import numpy as np 
# branch = ['CSE','CE','EC','ME']
# student_count = [100,30,50,45]
# index = np.arange(len(branch))
# plt.xlabel('BRANCH')
# plt.ylabel('COUNT OF STUDENTS')
# plt.xticks(index,branch,rotation=60)
# plt.bar(index,student_count,color = ['blue','red','cyan','yellow'])
# # plt.colorbar(label='Color Scale')
# plt.show()

# df = pd.read_csv('C:/44/pandas aiml/titanic.csv',index_col=0)



#chatgpt 
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the Titanic dataset
# df = pd.read_csv('C:/44/pandas aiml/titanic.csv', index_col=0)

# # Count the number of passengers in each Pclass
# pclass_counts = df['Pclass'].value_counts().sort_index()

# # Create the bar graph
# plt.bar(pclass_counts.index.astype(str), pclass_counts.values, color='skyblue', edgecolor='black')

# # Add title and axis labels
# plt.title("Number of Passengers by Passenger Class (Pclass)", fontsize=14)
# plt.xlabel("Passenger Class (Pclass)")
# plt.ylabel("Number of Passengers")

# # Add gridlines to y-axis
# plt.grid(axis='y', linestyle='--', linewidth=0.7)

# # Add count labels on top of each bar
# for i, val in enumerate(pclass_counts.values):
#     plt.text(i, val + 5, str(val), ha='center', fontsize=10)

# # Adjust layout and display the plot
# plt.tight_layout()
# plt.show()

#seaborn
# import seaborn as sns
# df = pd.read_csv("C:/44/pandas aiml/Toyota.csv")
# # df1 = sns.load_dataset('titanic')
# df.dropna(inplace=True)

# # sns.regplot(x=df['Age'],y=df['Price'],marker='D',fit_reg=True,color=['red','green'])
# sns.regplot(x='Age',y='Price',data=df,marker='D',fit_reg=True,color='red',scatter=True,ci=100,order=3,scatter_kws={'s':100,'alpha':0.6},line_kws={'linewidth':'3','color':'green'})
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample Data
# df = pd.DataFrame({
#     'Age': [1, 2, 3, 4, 5, 6, 7],
#     'Price': [20000, 19500, 18000, 17500, 16000, 15000, 14000]
# })

# # Basic Plot
# sns.regplot(x="Age", y="Price", data=df)
# plt.title("Car Price vs Age")
# plt.show()
# sns.set_style("whitegrid")      # Adds grid on white background
# sns.set_context("talk")         # Enlarges font & labels for presentation

# sns.regplot(x="Age", y="Price", data=df,
#             marker="*", ci=95,
#             line_kws={"color": "black", "linewidth": 2},
#             scatter_kws={"color": "red", "alpha": 0.6})

# plt.title("Car Price vs Age", fontsize=18)
# plt.xlabel("Car Age (Years)")
# plt.ylabel("Price (in Rs)")
# plt.grid(True)
# plt.show()
# styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]
# contexts = ["paper", "notebook", "talk", "poster"]

# for style in styles:
#     for context in contexts:
#         sns.set_style(style)
#         sns.set_context(context)

#         sns.regplot(x="Age", y="Price", data=df)
#         plt.title(f"Style: {style}, Context: {context}")
#         plt.show()

# hisrtogram , with ksde

# df = pd.read_csv("C:/44/pandas aiml/Toyota.csv")
# df.dropna(inplace=True)
# sns.distplot(df['Price'],
#              bins=2,
#              color='green',
#              kde=True,
#              rug=True,
#              hist_kws={'alpha': 0.5},
#              kde_kws={'linewidth': 2, 'color': 'blue'})





# Import libraries
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Generate sample DataFrame
# df = pd.DataFrame({
#     'department': np.random.choice(['CS', 'EC', 'Civil', 'Mechanical'], size=100),  # department: Randomly chosen from four categories
#     'salary': np.random.randint(25000, 100000, size=100),  # salary: Random integers between 25000 and 100000
#     'experience': np.random.randint(1, 20, size=100),  # experience: Random integers between 1 and 20 years
#     'gender': np.random.choice(['Male', 'Female'], size=100),  # gender: Randomly chosen Male or Female
#     'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Bangalore'], size=100)  # city: Randomly chosen from four cities
# })

# # 1️⃣ COUNT PLOT
# sns.countplot(
#     x='department',  # x-axis categories
#     data=df,  # source DataFrame
#     hue='gender',  # subgroup bars by gender
#     palette='pastel',  # set color palette
#     order=['CS', 'EC', 'Civil', 'Mechanical']  # define order of categories on x-axis
# )
# plt.title("Department Distribution by Gender")
# plt.show()

# # 2️⃣ BOX PLOT
# sns.boxplot(
#     x='department',  # x-axis categories
#     y='salary',  # y-axis numerical variable
#     hue='gender',  # split boxes within each department by gender
#     data=df,  # source DataFrame
#     palette='coolwarm',  # color palette
#     linewidth=2,  # thickness of box borders
#     width=0.5,  # width of each box
#     fliersize=4,  # size of outlier points
#     whis=1.5  # whisker length multiplier (1.5 * IQR)
# )
# plt.title("Salary Distribution Across Departments by Gender")
# plt.show()

# # 3️⃣ SCATTER PLOT
# sns.scatterplot(
#     x='salary',  # x-axis numerical variable
#     y='experience',  # y-axis numerical variable
#     hue='department',  # color points based on department
#     style='gender',  # marker style based on gender
#     size='experience',  # size of points based on experience
#     sizes=(20,200),  # range of point sizes
#     palette='deep',  # color palette
#     data=df  # source DataFrame
# )
# plt.title("Scatter Plot of Salary vs Experience")
# plt.show()

# # 4️⃣ HISTOGRAM (Matplotlib)
# plt.hist(
#     df['salary'],  # data to plot
#     bins=15,  # number of bins
#     color='purple',  # bar color
#     edgecolor='black',  # color of bar edges
#     alpha=0.7  # transparency
# )
# plt.title("Salary Distribution Histogram")
# plt.xlabel("Salary")
# plt.ylabel("Frequency")
# plt.show()

# # 5️⃣ HISTOGRAM (Seaborn)
# sns.histplot(
#     df['salary'],  # data to plot
#     bins=20,  # number of bins
#     color='green',  # bar color
#     kde=True  # include Kernel Density Estimate curve
# )
# plt.title("Salary Distribution with KDE")
# plt.show()

# # 7️⃣ VIOLIN PLOT
# sns.violinplot(
#     x='department',  # x-axis categories
#     y='salary',  # y-axis numerical variable
#     hue='gender',  # split violins within each department by gender
#     data=df,  # source DataFrame
#     split=True,  # split violins for each hue level
#     palette='muted'  # color palette
# )
# plt.title("Violin Plot of Salary Distribution by Department and Gender")
# plt.show()

# # 8️⃣ PAIR PLOT
# sns.pairplot(
#     df,  # source DataFrame
#     hue='department',  # color points by department
#     palette='Set1',  # color palette
#     corner=True,  # plot only lower triangle
#     diag_kind='kde',  # diagonal plots are Kernel Density Estimates
#     kind='scatter',  # off-diagonal plots are scatter plots
#     plot_kws={'alpha':0.6, 's':40},  # transparency and size of scatter points
#     diag_kws={'shade':True}  # shade area under KDE curve
# )
# plt.suptitle("Pair Plot of Dataset", y=1.02)
# plt.show()

# # 9️⃣ REGRESSION PLOT (regplot)
# sns.regplot(
#     x='experience',  # x-axis numerical variable
#     y='salary',  # y-axis numerical variable
#     data=df,  # source DataFrame
#     scatter_kws={'color':'blue'},  # set scatter point color
#     line_kws={'color':'red'}  # set regression line color
# )
# plt.title("Regression Line of Experience vs Salary")
# plt.show()


#  piechart
# departments = ['education','healthcare','defence','infrastructure']
# allocations = [25,30,20,25]
# stream = ['Science','Commerce','Arts','cyber']
# Enr = [45,35,10,10]
# job = ["Dev","QA","Designers","Admin"]
# dist = [50,20,15,15]
# meth = ["Credit","UPI","Cash","Wallets"]
# use = [45,35,10,10]
# disease = ["Viral","Bacterial","Chronic","Other"]
# prop = [40,35,15,10]

# series = pd.Series(departments)
# print(series)
# data = {"stream":stream,"enrollments":Enr}
# frame = pd.DataFrame(data)
# print(frame)

# # departmets and allocation
# plt.pie(x=allocations,labels=departments,colors=['red', 'green', 'blue', 'yellow'],        startangle=90,
#     autopct='%1.1f%%',
#     shadow=True,
#     explode= [0.1,0.2,0.3,0.4],
#     counterclock=True,
#     wedgeprops={'edgecolor': 'black', 'linewidth': 2},
#     textprops={'fontsize': 12, 'color': 'darkblue'},
#     radius=1,
#     center=(0,0),
#     frame=True,
#     pctdistance=0.7)
# plt.show()

# # #stream and enrollement
# plt.pie(x=Enr,labels=stream,colors=['cyan', 'magenta', 'black', 'white'],        startangle=90,
#     autopct='%1.1f%%',
#     shadow=True,
#     explode= [0.1,0.2,0.3,0.4],
#     counterclock=True,
#     wedgeprops={'edgecolor': 'black', 'linewidth': 2},
#     textprops={'fontsize': 12, 'color': 'darkblue'},
#     radius=1,
#     center=(0,0),
#     frame=True,
#     pctdistance=0.7)
# plt.show()

# # #jobs and distribution
# plt.pie(x=dist,labels=job,colors=['gray', 'lime', 'gold', 'salmon'],
#          startangle=90,
#     autopct='%1.1f%%',
#     shadow=True,
#     explode= [0.1,0.2,0.3,0.4],
#     counterclock=True,
#     wedgeprops={'edgecolor': 'black', 'linewidth': 2},
#     textprops={'fontsize': 12, 'color': 'darkblue'},
#     radius=1,
#     center=(0,0),
#     frame=True,
#     pctdistance=0.7)
# plt.show()

# # #methods and users 
# plt.pie(x=use,labels=meth,colors=['black','brown','orange','y'],
#          startangle=90,
#     autopct='%1.1f%%',
#     shadow=True,
#     explode= [0.1,0.2,0.3,0.4],
#     counterclock=True,
#     wedgeprops={'edgecolor': 'black', 'linewidth': 2},
#     textprops={'fontsize': 12, 'color': 'darkblue'},
#     radius=1,
#     center=(0,0),
#     frame=True,
#     pctdistance=0.7)
# plt.show()

# #disease and propogations 
# plt.pie(x=prop,labels=disease,colors=['r','c','g','b'],  
#         startangle=90,
#     autopct='%1.1f%%',
#     shadow=True,
#     explode= [0.1,0.2,0.3,0.4],
#     counterclock=True,
#     wedgeprops={'edgecolor': 'black', 'linewidth': 2},
#     textprops={'fontsize': 12, 'color': 'darkblue'},
#     radius=1,
#     center=(0,0),
#     frame=True,
#     pctdistance=0.7
#     )
# plt.show()


import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 


# df = sns.load_dataset('titanic')
# print(df.info())
# print(df.head(10))
# df.dropna(inplace=True)
# plt.bar(df['class'].sort_values().unique(),df['class'].value_counts(),
#             linewidth=2,               # Border thickness
#     align='center',            # Align bars at center
#     alpha=0.9,                 # Transparency
#     label='CLASS',     # Label for legend
#     zorder=3,                  # Draw on top of grid
#     hatch='//'                 # Stripe pattern fill)
# )

# plt.show()

data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Product A': [100, 110, 130, 125, 140, 150],
    'Product B': [120, 115, 125, 110, 130, 135],
    'Product C': [90, 95, 105, 100, 120, 125]
}

df = pd.DataFrame(data)
# # product1
# plt.plot(df['Month'],df['Product A'],c='red',label="Product A",marker='^')


# # product2
# plt.plot(df['Month'],df['Product B'],c='green',label="Product B",marker='^')


# # product3
# plt.plot(df['Month'],df['Product C'],c='blue',label="Product C",marker='^')
# plt.xlabel("MONTHS")
# plt.ylabel("PRODUCTS")
# plt.legend()
# plt.show()

# Create 2 subplots (1 row, 2 columns)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# # Plotting on the first axis
# axes[0].plot(df['Month'],df['Product A'],label='product A',marker='^')
# axes[0].set_title("Plot 1")

# # Plotting on the second axis
# axes[1].plot(df['Month'],df['Product B'])
# axes[1].set_title("Plot 2")

# # Plotting on the thrid axis
# axes[2].plot(df['Month'],df['Product B'])
# axes[2].set_title("Plot 2")

# plt.tight_layout()
# plt.show()

# fig_1 = plt.figure(figsize=(5,3))

# fig = plt.figure(figsize=(10, 3))
# ax1 = fig.add_subplot(1, 3, 1)
# ax2 = fig.add_subplot(1, 3, 2)
# ax3 = fig.add_subplot(1, 3, 3)
# plt.show()

# fig = plt.figure(figsize=(10, 5))
# subfigs = fig.subfigures(1, 2)

# axsLeft = subfigs[0].subplots(2, 1)
# axsRight = subfigs[1].subplots(1, 2)

# axsLeft[0].plot([1, 2, 3], [4, 5, 6])
# axsRight[0].bar([1, 2, 3], [7, 8, 5])
# plt.show()