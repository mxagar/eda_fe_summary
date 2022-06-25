'''This python script summarizes many code snippets for data processing.
The file won't run! It is a compilation of important code pieces ready
for copy & paste.

See the companion README.md with schematic explanations.

Table of contents:

- Imports
- General, Useful & Important Functions
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Inferences & Hypothesis Testings
- Data Modelling

Author: Mikel Sagardia.
Date: 2016 - 2022.
Repository: https://github.com/mxagar/eda_fe_summary

'''

### -- 
### -- Imports
### -- 

# Choose the imports that are necessary

import datetime as dt
import joblib # save python objects (warning: python version specific)

import pandas as pd
import numpy as np 
from scipy.stats import stats
import scipy.stats.distributions as dist

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

# Binary or one-hot encoding with Scikit-Learn
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
#from pandas import get_dummies

# Ordinal encoding with Scikit-Learn; ordinal = there is a ranking order between category levels
# Beware: we assume distances between categories are as encoded!
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OrdinalEncoder

# Lasso regression is the model used (L1 regularization: penaly to variable coefficients)
# SelectFromModel identifies the features that are important for the model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

##### -- 
##### -- General, Useful & Important Functions
##### -- 

df = pd.concat([X,y],axis=1)
df.to_csv('data/dataset.csv',sep=',', header=True, index=False)
df = pd.read_csv('data/dataset.csv')

df.head(3)
df.info()
df["price"].describe() # use .T if many columns

# Always save lists of numerical/continuous and categorical columns
categorical_cols = list(df.select_dtypes(include = ['object']))
numerical_cols = list(df.select_dtypes(include = ['float', 'int']))

# IQR = 1.5*(q75-q25) -> outlier detection
q25, q50, q75 = np.percentile(df['price'], [25, 50, 75])
# Skewness: a absolute value larger than 0.75 requires transformations
df['price'].skew()

# Get uniques and counts of the levels of a categorcial variable
df["condition"].value_counts().sort_values(ascending=False).plot(kind='bar')
df["condition"].unique()

# Do you need sampling?
# Pandas sample of n=5 rows/data-points that only appear once
sample = df.sample(n=5, replace=False)

# Basic text processing
word_list = text.lower().split(' ')
number = int(text.split(' ')[0])

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

# Pandas slicing:
# - `df[]` should access only to column names/labels: `df['col_name']`.
# - `df.iloc[]` can access only to row & column index numbers + booleans:
#		`df.iloc[0,'col_name']`, `df.iloc[:,'col_name']`.
# - `df.loc[]` can access only to row & column labels/names + booleans:
#		`df.loc['a','col_name']`, `df.loc[:,'col_name']`.
# - `df[]` can be used for changing entire column values,
#		but `df.loc[]` or `df.iloc[]` should be used for changing sliced row values.
df.loc[['a', 'b'],'BMXLEG'] # df.index = ['a', 'b', ...]
df.loc[[0, 1],3]

# Selection / Filtering: Column selection after name
col_names = df.columns # 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXLEG', 'BMXARML', ...
keep = [column for column in col_names if 'BMX' in column]
df_BMX = df[keep]

# Selection / Filtering: With booleans
df.loc[:, keep].head()
# # Indexing with booleans: Which column (names) are in keep?
index_bool = np.isin(df.columns, keep)
df_BMX = df.iloc[:,index_bool]

# Multiple filtering: Several conditions
waist_median = df_BMX['BMXWAIST'].median()
condition1 = df_BMX['BMXWAIST'] > waist_median
condition2 = df_BMX['BMXLEG'] < 32
df_BMX[condition1 & condition2].head()

# Multiple filtering: Another example
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
total_select = total.head(20)/df.shape[0]
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

# All rows with NA in column "variable" dropped, but not inplace!
df.dropna(subset=["variable"])

# Entire column "variable" dropped, but not inplace!
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

# Detect a word and encode
df['bath_private'] = df['bath_description'].apply(lambda text: 1 if 'private' in text else 0)

# Map the values == Replace
dictionary = {'value1':'val1', 'value3':'val2'}
df['variable'] = df['variable'].map(dictionary)

# Convert the dates to days since today
today = dt.datetime(2022,6,17)
for col in dat_cols:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')`# format='%d/%m/%Y', etc. 
    df[col] = df[col].apply(lambda col: int((today-col).days))

##### -- 
##### -- Exploratory Data Analysis (EDA)
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

# Scatterplots
df.plot.scatter(x='var1', y='var2')
plt.scatter(x=df['var1'], y=df['var2'])
sns.scatterplot(x="seniority",y="income",hue="gender",data=df)

# Jointplots: density isolines for a pair of quantitative variables
# Two variables plotted; type of scatterplot scecified + density histograms
sns.jointplot(x=df['var1'],y=df['var2'],kind='hex')

# Scatterplot -> Scatterplot + Linear regression line plotted
sns.lmplot(x='x', y='y', data=df)
sns.reglot(x='x', y='y', data=df)
# ... with hue
sns.lmplot(x='x', y='y', data=df,
           fit_reg=False, # No regression line
           hue='category') # Color by category group

# Boxplots
df.boxplot(by='category');
sns.boxplot(x="species", y="height", data=df)
sns.boxplot(data=df.loc[:, ["A", "B", "C", "D"]]) # just the boxes of A, B, C, D

# Boxplots: Nicer - points displayed
for var in categorcial_selected:
	# Boxplot
    sns.catplot(x=var, y='price', data=df, kind="box", height=4, aspect=1.5)
    # Data points
    sns.stripplot(x=var, y='price', data=df, jitter=0.1, alpha=0.3, color='k')

# Boxplots: stratify + apply hue
df["age_group"] = pd.cut(df.age, [18, 30, 40, 50, 60, 70, 80])
sns.boxplot(x="age_group", y="blood_pressure", hue="gender", data=df)

# Swarmplot (<- Boxplots): point swarms instead of boxes
sns.swarmplot(x="x", y="y", data=df)

# Violin plots (<- Boxplots): A box plot with varying box width
sns.violinplot(x="gender", y="age", data=df)

# Correlations and heatmaps
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn')
df.corr()['target'].sort_values(ascending=True).plot(kind='bar')

# Frequency tables: Stratify & Group
df["age_group"] = pd.cut(df.age, [18, 30, 40, 50, 60, 70, 80])
# Eliminate rare/missing values
dx = df.loc[~df.education_level.isin(["Don't know", "Missing"]), :] 
dx = dx.groupby(["age_group", "gender"])["education_level"]
dx = dx.value_counts()
# Restructure the results from 'long' to 'wide'
dx = dx.unstack()
# Normalize within each stratum to get proportions
dx = dx.apply(lambda x: x/x.sum(), axis=1)
# Limit display to 3 decimal places
print(dx.to_string(float_format="%.3f"))
# education_level        9-11    <9  College  HS/GED  Some college/AA
# age_group   gender                                              
# (18, 30] 	  Female    0.080 0.049    0.282   0.215            0.374
#             Male      0.117 0.042    0.258   0.250            0.333
# (30, 40]    Female    0.089 0.097    0.314   0.165            0.335
# ...

# Bars, Horizontal bars
plt.bar(np.arange(10), df.var.iloc[:10])
plt.barh(np.arange(10), df.var.iloc[:10])
sns.barplot(data=df,x='married',y='income',hue='gender')

# Countplots: bar plot of the selected variable
sns.countplot(x='gender', data=df)

# Pairplots: scatterplot/histogram matrix
# All variables plotted against each other: scatterplots, histograms
# If many variables, select the ones we're interested in
# Hue: separate/group by categories
sns.pairplot(df[selected], hue='species', size=3)

# FacetGrid: map plot types to a grid of plots
# 1: specify dataset and col groups/levels
plot = sns.FacetGrid(df, col='species', margin_titles=True)
# 2: which variable to plot in cols, and which plot type (hist)
plot.map(plt.hist, 'height', color='green')


##### -- 
##### -- Feature Engineering
##### -- 

# Always make a copy!
df = data.copy()

# Statsmodels: Yeo-Johnson and Box-Cox transormations
# Always store params!
df[var+'_box'], param_box = stats.boxcox(df[var])
df[var+'_yeo'], param_yeo = stats.yeojohnson(df[var])
# We can do it in a for-loop for selected variables + visualize changes
for var in selected_cols:
    tmp[var], param = stats.yeojohnson(df[var])    
    plt.subplot(1, 2, 1)
    plt.scatter(df[var], df['target'])
    plt.ylabel('Target')
    plt.xlabel('Original ' + var)
    plt.subplot(1, 2, 2)
    plt.scatter(tmp[var], df['target'])
    plt.ylabel('Target')
    plt.xlabel('Transformed ' + var)

# Scikit-Learn Power Transformer
pt = PowerTransformer(method='box-cox', standardize=False) # method=‘yeo-johnson’
df['var_transformed'] = pt.fit_transform(df['var'])

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

# One-hot encoding: Dummy variables with pandas
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

# Check sparsity with PCA
scaler = StandardScaler()
X = scaler.fit_transform(X.astype(np.float64))
# PCA with all components
pca = PCA(n_components = X.shape[1])
pca.fit_transform(x)
# Explained variance of each principal component
explained_variance = pca.explained_variance_ratio_
# Sum all explained variances until 95% is reached;
# how many components do we need?
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >=0.95) + 1
# Another way to answer the question above:
# pass varinace ratio float as n_components
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
X_reduced.shape[1]



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

def z_test(subset1, subset2):
    '''This function computes the Z Test of two population proportions.
    H0: The proportions are the same.
    Ha: The proportions are different (two-sided).
    
    Input:
        subset1: data frame with values to analyze for subset/group 1
        subset1: data frame with values to analyze for subset/group 2
    Output:
        z_stat: Z statistic; if very negative, subset1 has a considerebly smaller proportion
        p_value: if < 0.05, the difference is significant, the two proportions are different
    '''
    # Sample sizes
    n1 = subset1.shape[0]
    n2 = subset2.shape[0]
    
    # Number of positive values
    y1 = subset1.sum()
    y2 = subset2.sum()

    # Estimates of the population proportions
    p1 = np.round(y1 / n1, 2)
    p2 = np.round(y2 / n2, 2)
    
    if p1 == p2:
        print('Warning: same proportions!')

    # Estimate of the combined population proportion
    phat = (y1 + y2) / (n1 + n2)

    # Estimate of the variance of the combined population proportion
    va = phat * (1 - phat)

    # Estimate of the standard error of the combined population proportion
    se = np.sqrt(va * ((1.0 / n1) + (1.0 / n2)))

    # Test statistic and its p-value: 2-sided, because the Ha is !=
    z_stat = (p1 - p2) / se
    p_value = 2*dist.norm.cdf(-np.abs(z_stat))
    # Equivalent to
    # p_value = 2*(1-dist.norm.cdf(np.abs(z_stat)))

    return (z_stat, p_value, p1, p2)

# Apply Z Test to binary variables (i.e., proportions of groups)
group1 = 'Donostia'
group2 = 'Bilbao'
feature_analyze = 'bathrooms_shared'
subset1 = df[df[group1] > 0][feature_analyze]
subset2 = df[df[group2] > 0][feature_analyze]
(z_stat, p_value, p1, p2) = z_test(subset1, subset2)

def t_test(subset1, subset2):
    '''This function computes the Z Test of two population proportions.
    H0: The proportions are the same.
    Ha: The proportions are different (two-sided).

    Input:
        subset1: data frame with values to analyze for subset/group 1
        subset1: data frame with values to analyze for subset/group 2
    Output:
        t_stat: T statistic; if very negative, subset1 has a considerebly smaller mean
        p_value: if < 0.05, the difference is significant, the two distribution means are different
    '''
    
    # Sample sizes    
    n1 = subset1.shape[0]
    n2 = subset2.shape[0]
    
    # Means
    m1 = subset1.mean()
    m2 = subset2.mean()

    # Standard Deviations
    sd1 = subset1.std()
    sd2 = subset2.std()

    # Standard Error
    if sd1/sd2 < 2.0 or sd2/sd1 < 2.0:
        # Similar variances: Pooled
        se = np.sqrt(((sd1**2)/n1) + ((sd2**2)/n2))
    else:
        # Different variances: Unpooled
        se = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2)/(n1+n2-2)) * np.sqrt((1/n1) + (1/n2))
    
    # T statistic
    t_stat = (m1 - m2)/se    
    # With T and df, we can get the p-value from the T distribution
    df = n1 + n2 - 2
    # p-value is obtained form the cummulative density function (CDF) with the given df
    # since we want the remaining are under the PDF, we need to compute 1-CDF(t)
    # Since it is a 2-sided test (Ha !=), we need to x2 the p-value
    p_value = 2*(1 - dist.t.cdf(np.abs(t_stat),df=df))
    
    return (t_stat, p_value, m1, m2)

# Apply T Test to continuous variables (means of groups)
group1 = 'has_beach' # == 1
group2 = 'has_beach' # == 0
feature_analyze = 'price'
subset1 = df[df[group1] > 0][feature_analyze]
subset2 = df[df[group2] < 1][feature_analyze]
(t_stat, p_value, m1, m2) = t_test(subset1, subset2)


##### -- 
##### -- Data Modelling
##### -- 

