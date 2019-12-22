# --------------
# Code starts here
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from scipy.stats import skew
#### Data 1
# Load the data
df= pd.read_csv(path)

# Overview of the data
df.info()

# Histogram showing distribution of car prices
plt.figure()
sns.distplot(df['price'],kde=True,rug=True)

# Countplot of the make column
plt.figure()
sns.countplot(y='make',data=df)

# Jointplot showing relationship between 'horsepower' and 'price' of the car
plt.figure()
sns.jointplot('horsepower','price',data=df,kind='reg')

# Correlation heat map
plt.figure(figsize=(12,12))
sns.heatmap(data=df.corr(),cmap="YlGnBu")
# boxplot that shows the variability of each 'body-style' with respect to the 'price'
plt.figure()
sns.boxplot(x="body-style",y='price',data=df)

#### Data 2

# Load the data
df1=pd.read_csv(path2)
print(df1.shape)
print(df1.columns)
# Impute missing values with mean
df1 = df1.replace("?","NaN")
print(df1.isna().sum())
mean_imputer = Imputer(strategy='mean')
df1[['normalized-losses']]=mean_imputer.fit_transform(df1[['normalized-losses']])
df1[['horsepower']]=mean_imputer.fit_transform(df1[['horsepower']])

# Skewness of numeric features
numeric_columns=df1._get_numeric_data().columns
print(numeric_columns)
for i in numeric_columns:
    if skew(df1[i]) >1:
        df1[i] = np.sqrt(df1[i])
# Label encode 
categorical_col=df1.select_dtypes(include='object').columns
print(categorical_col)
print(df1[categorical_col].head(n=5))
encoder= LabelEncoder()
for i in categorical_col:
    df1[i]=encoder.fit_transform(df1[i])
print(df1[categorical_col].head(n=5))
df1['area']=df1['height']*df1['width']
# Code ends here


