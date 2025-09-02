import pandas as pd 
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA