import pandas as pd 
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay
from sklearn.linear_model import LogisticRegression 


dataset = load_breast_cancer()
x,y = dataset.data , dataset.target

print("features",dataset.feature_names)
print("target class",dataset.feature_names)