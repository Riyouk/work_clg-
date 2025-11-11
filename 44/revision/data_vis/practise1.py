# =============================================
# 📊 DATA VISUALIZATION USING PYTHON - COMPLETE GUIDE
# =============================================

# 1️⃣ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# =============================================
# 2️⃣ BASIC VISUALIZATIONS - MATPLOTLIB
# =============================================

# Line Plot
x = np.arange(0, 10, 0.5)
y = np.sin(x)
plt.figure(figsize=(6,4))
plt.plot(x, y, color='blue', marker='o', linestyle='--', linewidth=2)
plt.title('Line Plot Example')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.grid(True)
plt.show()

# Bar Plot
categories = ['A', 'B', 'C', 'D']
values = [10, 23, 15, 7]
plt.bar(categories, values, color='orange')
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()

# Horizontal Bar Plot
plt.barh(categories, values, color='green')
plt.title('Horizontal Bar Plot')
plt.show()

# Scatter Plot
x = np.random.randint(10, 100, 50)
y = x + np.random.randint(-10, 10, 50)
plt.scatter(x, y, color='purple')
plt.title('Scatter Plot Example')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Pie Chart
sizes = [40, 30, 20, 10]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Pie Chart Example')
plt.show()

# Box Plot
data = np.random.normal(100, 20, 200)
plt.boxplot(data)
plt.title('Box Plot Example')
plt.show()

# =============================================
# 3️⃣ ADVANCED VISUALIZATIONS - SEABORN
# =============================================

# Heatmap
data = np.random.rand(5, 5)
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title('Heatmap Example')
plt.show()

# Pairplot (using Iris Dataset)
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species')
plt.suptitle('Pairplot - Iris Dataset', y=1.02)
plt.show()

# Countplot
sns.countplot(x='species', data=iris)
plt.title('Count of Each Species')
plt.show()

# =============================================
# 4️⃣ CUSTOMIZATION AND STYLING
# =============================================

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='red', label='sin(x)')
plt.title('Customized Line Plot')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.grid(True)
plt.show()

# Change Seaborn Style
sns.set_style("darkgrid")
sns.histplot(np.random.randn(100), kde=True, color='purple')
plt.title('Seaborn Style Example')
plt.show()

# =============================================
# 5️⃣ REAL DATASET VISUALIZATION (TIPS DATA)
# =============================================

tips = sns.load_dataset('tips')

# 1. Bar plot - Average tip by day
sns.barplot(x='day', y='tip', data=tips, palette='viridis')
plt.title('Average Tip per Day')
plt.show()

# 2. Scatter plot - Total bill vs Tip
sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)
plt.title('Total Bill vs Tip (by Gender)')
plt.show()

# 3. Box plot - Tips distribution per Day
sns.boxplot(x='day', y='tip', data=tips)
plt.title('Tip Distribution by Day')
plt.show()

# 4. Heatmap - Correlation Matrix
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Tips Dataset)')
plt.show()

# =============================================
# 6️⃣ MULTIPLE PLOTS SIDE BY SIDE
# =============================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.barplot(x='day', y='tip', data=tips)
plt.title('Bar Plot')

plt.subplot(1,2,2)
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Scatter Plot')

plt.tight_layout()
plt.show()

# =============================================
# 7️⃣ COMPARISON SUMMARY
# =============================================

print("""
=============================================
📘 DATA VISUALIZATION SUMMARY
=============================================
✅ Matplotlib → Base library, full control
✅ Seaborn → High-level, beautiful statistical plots
✅ Plotly → Interactive dashboards (optional next step)
✅ Pandas .plot() → Quick visualization from DataFrames

Next Recommended Topics:
1. Interactive Visuals with Plotly/Dash
2. Streamlit Dashboards
3. EDA using Pandas & Seaborn
4. Geospatial Visualization with Folium
=============================================
""")
