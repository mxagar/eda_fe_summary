'''This python script summarizes many code snippets for data processing.
The file won't run! It is a compilation of important code pieces ready
for copy & paste.

Table of contents:

'''

### -- Imports

import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import stats

# Display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

### -- Useful Functions

df.to_csv('data/dataset.csv',sep=',',header=True,index=False)
df = pd.read_csv('data/dataset.csv')

df.head(3)
df.info()
df["price"].describe()

# Always save lists of numerical/continuous and categorical columns
categorical_cols = list(df.select_dtypes(include = ['object']))
numerical_cols = list(df.select_dtypes(include = ['float', 'int']))

# IQR = 1.5*(q75-q25) -> outlier detection
q25, q50, q75 = np.percentile(df['price'], [25, 50, 75])
# Skewness: 
df['price'].skew()

# Get counts of the levels of a categorcial variable
df["condition"].value_counts()

### -- Data Cleaning

# Get duplicated rows
# df.duplicated(['id']) -> False, False, ...
duplicate = df[df.duplicated(['id'])]
# Drop duplicates
duplicated_removed = df.drop_duplicates()
# Check that all indices are unique
df.index.is_unique

# Detect missing values, sort them ascending, plot
# isnull() == isna()
total = df.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

# All rows with NA in column "Variable" dropped, but not inplace!
df.dropna(subset=["Variable"])

# Entire column "Lot Frontage" dropped, but not inplace!
df.drop("Variable", axis=1)

# Compute median of a column/feature
median = df["Variable"].median() # also: mean(), std(), mode(), etc.
# Replace/impute NA values with median
df["Variable"].fillna(median, inplace = True)



### -- Exploratory Data Analysis

# Histograms
plt.figure(figsize=(10,4))
sns.histplot(df[df.has_beach == 1].price,bins=80,color='blue',alpha=0.5)
sns.histplot(df[df.has_beach == 0].price,bins=80,color='red',alpha=0.5)
plt.legend(['Beach < 2km', 'Beach > 2km'])
plt.xlabel('Price, USD')
plt.title('Price distribution depending on access to beach')
plt.savefig('./pics/price_distribution_beach.png',dpi=300,transparent=False,bbox_inches='tight')

### -- Feature Engineering

boxcox_transformed = pd.Series(stats.boxcox(df['SalePrice'])[0])



### -- Feature Selection


### -- Inferences & Hypothesis Testings


### -- Data Modelling

