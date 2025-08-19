import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import random 
sns.set_palette('deep')
# sns.set_theme('talk')
# sns.set
Age = np.random.randint(30,90,10000)
mean = []
for i in range(100):
    x = random.choices(Age,k=10)
    print(x)
    y = np.mean(x)
    print(y)
    mean.append(y)
    y = 0
print(mean)
fig,axis = plt.subplots(2,2,figsize=(18,9))
sns.histplot(x=Age,bins=10,kde=True,ax=axis[0,0])
axis[0,0].set_title("Original Age Distribution ")
axis[0,0].set_xlabel("AGE")
axis[0,0].set_gid(True)
sns.histplot(x=mean,bins=10,kde=True,ax=axis[0,1])
axis[0,1].set_title("Distribution of Sample Means (Central Limit Theorem Demo)")
axis[0,1].set_xlabel("AGE")
axis[0,1].set_gid(True)
# axis[0,1].set_facecolor('red')
print(Age)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

def mean_generation(): 
    df = pd.read_csv("C:/44/pandas aiml/retirement_age_dataset.csv")
    print(df.head())

    mean = []
    
    for i in range(100):
        # Sample from the 'Retirement_Age' column
        x = random.choices(df['Retirement_Age'], k=10)
        print(x)

        y = np.mean(x)
        print("Sample Mean:", y)

        mean.append(y)
    
    print("All Sample Means:", mean)

    # Plot histogram of original data
    # fig,ax = plt.subplots(1,2,figsize=(16,8))
    sns.histplot(x=df['Retirement_Age'], kde=True, bins=10,ax=axis[1,0])
    plt.title("Original Retirement Age Distribution")
    axis[1,0].set_xlabel("AGE")
    axis[1,0].set_gid(True)
    # plt.show()

    # Plot histogram of sample means
    sns.histplot(x=mean, kde=True, bins=10,ax=axis[1,1],)
    axis[1,1].set_xlabel("AGE")
    plt.title("Distribution of Sample Means (Central Limit Theorem Demo)")
    axis[1,1].set_gid(True)
    plt.tight_layout()
    plt.show()

# Run the function
mean_generation()

