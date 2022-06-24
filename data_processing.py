'''This python script summarizes many code snippets for data processing.
The file won't run! It is a compilation of important code pieces ready
for copy & paste.

Table of contents:

'''

### -- 
### -- Imports
### -- 

import datetime as dt
import joblib # save python objects

import pandas as pd
import numpy as np 
from scipy.stats import stats

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Lasso regression is the model used (L1 regularization: penaly to variable coefficients)
# SelectFromModel identifies the features that are important for the model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

##### -- 
##### -- Useful Functions
##### -- 

df = pd.concat([X,y],axis=1)
df.to_csv('data/dataset.csv',sep=',', header=True, index=False)
df = pd.read_csv('data/dataset.csv')

df.head(3)
df.info()
df["price"].describe()

# Always save lists of numerical/continuous and categorical columns
categorical_cols = list(df.select_dtypes(include = ['object']))
numerical_cols = list(df.select_dtypes(include = ['float', 'int']))

# IQR = 1.5*(q75-q25) -> outlier detection
q25, q50, q75 = np.percentile(df['price'], [25, 50, 75])
# Skewness: a absolute value larger than 0.75 requires transformations
df['price'].skew()

# Get uniques and counts of the levels of a categorcial variable
df["condition"].value_counts()
df["condition"].unique()

# Do you need sampling?
# Pandas sample of n=5 rows/data-points that only appear once
sample = df.sample(n=5, replace=False)

# GROUP BY
# When grouping by a column/field,
# we need to apply the an aggregate function
df.groupby('species').mean()
df.groupby('species').agg([np.mean, np.median])
df.groupby(['year', 'city'])['var'].median()

# Dates
df['date'] = pd.to_datetime(df['date'], format='%b-%y')
df['month'] = df['date'].dt.month_name().str.slice(stop=3)
df['year'] = df['date'].dt.year

# Multiple filtering
df_filtered = df[(df['location'] == "Munich, Germany") | (df['location'] == "London, England")]
cities = ['Munich', 'London', 'Madrid']
df_filtered = df[df.location.isin(cities)]


##### -- 
##### -- Data Cleaning
##### -- 

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
df.dropna(subset=["variable"])

# Entire column "Lot Frontage" dropped, but not inplace!
df.drop("variable", axis=1)

# Compute median of a column/feature
median = df["variable"].median() # also: mean(), std(), mode(), etc.
# Replace/impute NA values with median
df["variable"].fillna(median, inplace = True)

# Box plot: detect outliers that are outside the 1.5*IQR
# Keeping or removing them depends on our understanding of the data
# Try: boxplots, log transformation, scatterplot with target, Z score
sns.boxplot(x=df['variable'])
sns.boxplot(x=np.log(df['variable']))
df.plot.scatter(x='variable', y='price')
df['z_variable'] = stats.zscore(df['variable'])

# Manually removing using the index
outliers_dropped = df.drop(housing.index[[1499,2181]])

# Rename columns
df = df.rename(columns={'fam': 'family'})

# Replace category names/values
df['gender'] = df['gender'].replace({'male':1, 'female':0})
df['gender'].replace({'male':1, 'female':0}, inplace=True)

# Map the values == Replace
dictionary = {'value1':'val1', 'value3':'val2'}
df['variable'] = df['variable'].map(dictionary)

# Convert the dates to days since today
today = dt.datetime(2022,6,17)
for col in dat_cols:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')`# format='%d/%m/%Y', etc. 
    df[col] = df[col].apply(lambda col: int((today-col).days))

##### -- 
##### -- Exploratory Data Analysis
##### -- 

# Matplotlib: Several diagrams overlaid / Scatterplots if linestyle ''
fig = plt.figure(figsize=(10,10))
# Scatterplot, because ls=''
plt.plot(df.var1, df.var2, ls='', marker='o', color='red', label='Vars 1 & 2')
# Scatterplot, in same diagram
plt.plot(df.var3, df.var4, ls='', marker='o', color='red', label='Vars 3 & 4')
plt.set(xlabel='vars 1, 3')
plt.legend(['A', 'B'])
plt.ylabel('vars 2, 4')
plt.title('Title')
plt.savefig('./pics/scatterplors.png',dpi=300,transparent=False,bbox_inches='tight')
plt.show() # if in a script

# Matplotlib: Subplots
fig, axes = plt.subplots(nrows = 1, ncols = 2)
for ax in axes:
	ax.plot(x,y)

# Pandas plotting
df['language'].value_counts().plot(kind='bar')
plt.title('Languages')

# Histograms
plt.hist(df.var, bins=25)
sns.histplot(df.var, bins=25)

# Histograms: Nicer
plt.figure(figsize=(10,4))
sns.histplot(df[df.has_beach == 1].price, bins=80, color='blue', alpha=0.5)
sns.histplot(df[df.has_beach == 0].price, bins=80, color='red', alpha=0.5)
plt.legend(['Beach < 2km', 'Beach > 2km'])
plt.xlabel('Price, USD')
plt.title('Price distribution depending on access to beach')
plt.savefig('./pics/price_distribution_beach.png', dpi=300, transparent=False, bbox_inches='tight')

# Bars, Horizontal bars
plt.bar(np.arange(10), df.var.iloc[:10])
plt.barh(np.arange(10), df.var.iloc[:10])

# Scatterplots
df.plot.scatter(x='var1', y='var2')
plt.scatter(x=df['var1'], y=df['var2'])

# Boxplots
df.boxplot(by='category');

# Correlations and heatmaps
sns.heatmap(new_data.corr(),annot=True,cmap='RdYlGn')


# PAIRPLOT
# All variables plotted against each other: scatterplots, histograms
# If many variables, select the ones we're interested in
# Hue: separate/group by categories
sns.pairplot(df[selected], hue='species', size=3)

# JOINTPLOT
# Two variables plotted; type of scatterplot scecified + density histograms
sns.jointplot(x=df['var1'],y=df['var2'],kind='hex')

# FACETGRID: map plot types to a grid of plots
# 1: specify dataset and col groups/levels
plot = sns.FacetGrid(df, col='species', margin_titles=True)
# 2: which variable to plot in cols, and which plot type (hist)
plot.map(plt.hist, 'height', color='green')


##### -- 
##### -- Feature Engineering
##### -- 

# Always make a copy!
df = data.copy()


boxcox_transformed = pd.Series(stats.boxcox(df['SalePrice'])[0])

# Multiplicative interaction
df['v1_x_v2'] = df['v1'] * df['v2']
# Division interaction: watch out - division by zero?
df['v1_/_v2'] = df['v1'] / df['v2']

# Create categorical data from continuous it that has a meaning
df['daytime'] = pd.cut(df.hour, [0,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'])

# Polynomial features with sklearn
pf = PolynomialFeatures(degree=2)
features = ['var1', 'var2']
pf.fit(df[features])
feat_array = pf.transform(df[features])
df_poly = pd.DataFrame(feat_array, columns = pf.get_feature_names(input_features=features))
df = pd.concat([df,df_poly])

# Replace categorical levels with few counts with 'other'
nbh_counts = df.neighborhood.value_counts()
other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
df['neighborhood'] = df['neighborhood'].replace(other_nbhs, 'other')

# Create deviation features: apply `groupby` to a categorical variable
# and compute deviation factors of another variable within each group
def add_deviation_feature(df, feature, category):
    category_gb = X.groupby(category)[feature]
    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())    
    deviation_feature = (df[feature] - category_mean) / category_std 
    df[feature + '_dev_' + category] = deviation_feature 
add_deviation_feature(df, 'overall_quality', 'neighborhood')


# Dummies with pandas / One-hot encoding
col_dummies = ['var1', 'var2']
try:
    for col in col_dummies:
        df = pd.concat([df.drop(col, axis=1),
        				pd.get_dummies(df[col], prefix=col, prefix_sep='_',
        					drop_first=False, dummy_na=False)],
                        axis=1)
except KeyError as err:
    print("Columns already dummified!")


# Train/Test split
y = df['price']
X = df.drop('price',axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, # predictive variables
    y, # target
    test_size=0.1, # portion of dataset to allocate to test set
    random_state=42, # we are setting the seed here
)


# Fit the scaler to the train set only
scaler = MinMaxScaler()
scaler.fit(X_train) 

# Transform both train/split
# and convert the numpy arrays
# into data frames
X_train = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns
)

# Always save the scaler!
joblib.dump(scaler, filepath+'minmax_scaler.joblib')

##### -- 
##### -- Feature Selection
##### -- 

# Set and save the seed for reproducibility
sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=42))
# Train Lasso model and select features
sel_.fit(X_train, y_train)
# List of selected features
selected_features = list(X_train.columns[(sel_.get_support())])

# Save the seleceted features
joblib.dump(selected_features, filepath+'selected_features.joblib')

##### -- 
##### -- Inferences & Hypothesis Testings
##### -- 

##### -- 
##### -- Data Modelling
##### -- 

