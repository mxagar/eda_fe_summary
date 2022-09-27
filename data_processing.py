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
from scipy.stats import chi2_contingency

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm 

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, plot_roc_curve
from sklearn.metrics import precision_recall_fscore_support as classification_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Encoding with Scikit-Learn
from sklearn import preprocessing # LabelEncoder, LabelBinarizer, OneHotEncoder
# from pandas import get_dummies

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

# Serialize and save any python object, e.g., a model/pipeline
# BUT: be aware that python versions must be consistent
# when saving and loading.
import pickle
pickle.dump(model, open('model.pickle','wb')) # wb: write bytes
model = pickle.load(open('model.pickle','rb')) # rb: read bytes

df.head(3)
df.info()
df["price"].describe() # use .T if many columns
df.dtypes.value_counts() # counts of each type

# Always save lists of numerical/continuous and categorical columns
categorical_cols = list(df.select_dtypes(include = ['object']))
numerical_cols = list(df.select_dtypes(include = ['float', 'int']))
# Other ways:
categorical_cols = [var for var in df.columns if df[var].dtype == 'O']

# Create a dataframe which records the number of unique variables values
# for each variable
df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
categorical_variables = list(df_uniques[(6 >= df_uniques['Unique Values']) & (df_uniques['Unique Values'] > 2)].index)

# Cast a variable
df['var'] = df['var'].astype('O')

# IQR = 1.5*(q75-q25) -> outlier detection
q25, q50, q75 = np.percentile(df['price'], [25, 50, 75])
# Skewness: an absolute value larger than 0.75 requires transformations
df['price'].skew()

# Get uniques and counts of the levels of a categorcial variable
df["condition"].value_counts().sort_values(ascending=False).plot(kind='bar')
df["condition"].unique()

# Pandas sample of n=5 rows/data-points that only appear once
sample = df.sample(n=5, replace=False)

# Basic text processing
word_list = text.lower().split(' ')
number = int(text.split(' ')[0])

# Group By
# When grouping by a column/field,
# we apply the an aggregate function
df.groupby('species').mean()
df.groupby('species').agg([np.mean, np.median])
df.groupby(['year', 'city'])['var'].median()

# Average job satisfaction depending on company size
df.groupby('CompanySize')['JobSatisfaction'].mean().dropna().sort_values()

# Dates: are usually 'object' type, they need to be converted & processed
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
df.iloc[[0, 1],3]

# Selection / Filtering: Column selection after name
col_names = df.columns # 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXLEG', 'BMXARML', ...
keep = [column for column in col_names if 'BMX' in column]
df_BMX = df[keep]

# Selection / Filtering: With booleans
df.loc[:, keep].head()
# Indexing with booleans: Which column (names) are in keep?
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

# Plotting styles can be modified!
sns.set_context('talk')
sns.set_style('white')

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

# Columns/Feature with NO missing values
no_nulls = set(df.columns[df.isnull().sum()==0])
# Columns/Feature with more than 75% of values missing
most_missing_cols = set(df.columns[(df.isnull().sum()/df.shape[0]) > 0.75])

# Detect missing values, sort them ascending, plot
# isnull() == isna()
total = df.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)/df.shape[0]
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.axhline(y=0.90, color='r', linestyle='-') # 90% missing line
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

# Analyze the effect of missing values on the target:
# take a feature and compute the target mean & std. for two groups: 
# (1) missing feature, (2) non-missing feature.
def analyse_na_value(df, var):
	# IMPORTANT: copy
    df = df.copy()
    df[var] = np.where(df[var].isnull(), 1, 0)
    tmp = df.groupby(var)['target'].agg(['mean', 'std'])
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')
for var in vars_with_na:
    analyse_na_value(data, var)

# All rows with NA in column "variable" dropped, but not inplace!
df.dropna(subset=["variable"])
# Entire column "variable" dropped, but not inplace!
df.drop("variable", axis=1)
# More dropping options
all_drop  = df.dropna() # all rows with at least one NA columns dropped
all_row = df.dropna(how='all') # all rows with all cols NA dropped
only3or1_drop = small_dataset.dropna(subset=['col1','col3']) # all rows with at least one NA in col1 or col3 dropped
df.drop('C',axis=1,inplace=True) # drop complete column C in place; default axis = 0, ie., rows

# Compute median of a column/feature
median = df["variable"].median() # also: mean(), std(), mode(), etc.
# Replace / Impute NA values with median
df["variable"].fillna(median, inplace = True)

# Imputation: More options
fill_mode = lambda col: col.fillna(col.mode()[0]) # mode() returns a series, pick first value
df = df.apply(fill_mode, axis=0)
# BUT: Prefer better this approach
# because apply might lead to errors
num_vars = df.select_dtypes(include=['float', 'int']).columns
for col in num_vars:
    df[col].fillna((df[col].mean()), inplace=True)

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

# We can also map functions.
# Example: Take absolute correlations values 
abs_correlations = correlations.map(abs).sort_values()

# Tempodal data: Convert the dates to days since today
today = dt.datetime(2022,6,17)
for col in dat_cols:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')`# format='%d/%m/%Y', etc. 
    df[col] = df[col].apply(lambda col: int((today-col).days))


##### -- 
##### -- Exploratory Data Analysis (EDA)
##### -- 

# Matplotlib: Several diagrams overlaid
# plot(): point coordinates joined with lines; scatterplots if linestyle ''
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

# Pandas plotting; plt settings passed as argument
df['language'].value_counts().plot(kind='bar')
plt.title('Languages')

# Histograms
plt.hist(df.var, bins=25)
sns.histplot(df.var, bins=25)
df.plot.hist(bins=25)

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
# Two variables plotted; type of scatterplot specified + density histograms
sns.jointplot(x=df['var1'],y=df['var2'],kind='hex')

# Scatterplot + Linear regression line plotted
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
# cmap: https://matplotlib.org/stable/gallery/color/colormap_reference.html
# palette: https://seaborn.pydata.org/tutorial/color_palettes.html
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt=".2f")
df.corr()['target'].sort_values(ascending=True).plot(kind='bar')
# -
correlations = df[selected_fields].corrwith(y) # correlations with target array y
correlations.sort_values(inplace=True)

# Correlations if we have many numerical variables: we can't visualize the matrix
corr_values = df[numerical_cols].corr()
# tril_indices_from returns a tuple of 2 arrays:
# the arrays contain the indices of the diagonal + lower triangle of the matrix:
# ([0,1,...],[0,0,...])
tril_index = np.tril_indices_from(corr_values)
# Make the unused values NaNs
# NaN values are automatically dropped below with stack()
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
# Stack the data and convert to a data frame
corr_values = (corr_values
               .stack() # multi-index stacking of a matrix: [m1:(m11, m12,...), m2:(m21, m22,...), ...]
               .to_frame() # convert in dataframe
               .reset_index() # new index
               .rename(columns={'level_0':'feature1', # new column names
                                'level_1':'feature2',
                                0:'correlation'}))
# Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()
# Plot
corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
# Query the most highly correlated values
corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')

# Frequency tables: Stratify & Group
# Recipe: groupby(), value_counts(), normalize with apply().
# See also: pd.crosstab()
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
sns.barplot(data=df, x='married', ='income', hue='gender')

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

# Normality checks
# Even though it is not necessary to have a normally distributed target
# having it so often improves the R2 of the model.
# We can check the normality of a variable in two ways:
# - visually: hist(), QQ-plot
# - with normality checks, e.g., D'Agostino

# Normality check: D'Agostino
# if p-value > 0.05: normal;
# the larger the p-value, the larger the probability of normality
from scipy.stats.mstats import normaltest # D'Agostino K^2 Test
normaltest(df['target'].values)

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

# Check if there are inverse transformation functions!
# Box-Cox has one; you need the lambda parameter that was computed
from scipy.special import inv_boxcox
y_pred = inv_boxcox(y_pred_bc,lmbd)

# Scikit-Learn Power Transformer
# Box-Cox: all positive
# Yeo-Johnson: positive and negative
pt = PowerTransformer(method='box-cox', standardize=False) # method=‘yeo-johnson’
#df['var_transformed'] = pt.fit_transform(df['var'])
df['var_transformed'] = pt.fit_transform(df['var'].values.reshape(-1,1)) # we might need to reshape
pt.inverse_transform(df['var_transformed'].values.reshape(-1,1)) # we should get df['var']

# Multiplicative interaction
df['v1_x_v2'] = df['v1'] * df['v2']
# Division interaction: watch out - division by zero?
df['v1_/_v2'] = df['v1'] / df['v2']

# Create categorical data from continuous it that has a meaning
# Or when the numerical variable has a skewed/irregular distribution
df['daytime'] = pd.cut(df.hour, [0,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'])

# Polynomial features with sklearn
pf = PolynomialFeatures(degree=2, include_bias=False)
features = ['var1', 'var2']
pf.fit(df[features])
feat_array = pf.transform(df[features])
df_poly = pd.DataFrame(feat_array, columns = pf.get_feature_names_out(input_features=features))
df = pd.concat([df,df_poly])

# Measure the cardinality of the categorical variables:
# how many catgeories they have.
df[categorical_vars].nunique().sort_values(ascending=False).plot.bar(figsize=(12,5))

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

# Binarization: usually very skewed variables are binarized.
# We can do it with apply(), np.where() or some other ways.
# However, we should check the predictive strength of binarized variables
# with bar plots and T tests:
# we binarize and compute the mean & std. of the target
# according to the binary groups.
for var in skewed:
    # IMPORTANT: copy
    tmp = data.copy()
    tmp[var] = np.where(df[var]==0, 0, 1)
    tmp = tmp.groupby(var)['target'].agg(['mean', 'std'])
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

# Encoding of the target classes: LabelEncoder(), LabelBinarizer()
# LabelEncoder() converts class strings into integers,
# necessary for target values in multi-class classification
# LabelBinarizer() converts integer class values into one-hot encoded arrays
# necessary for multi-class target values if ROC needs to be computed
from sklearn import preprocessing
# Encode names as 0...(n-1) class numbers
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
list(le.classes_) # ['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"]) # [2, 2, 1]
list(le.inverse_transform([2, 2, 1])) # ['tokyo', 'tokyo', 'paris']
le.classes_ # we get the class names
# Encode class numbers as binary vectors
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
lb.classes_ # array([1, 2, 4, 6])
lb.transform([1, 6]) # array([[1, 0, 0, 0], [0, 0, 0, 1]])

# Note: we can apply any encoder to selected columns!
for column in binary_variables:
    df[column] = lb.fit_transform(df[column])

# Make a feature explicitly categorical (as in R)
# This is not necessary, but can be helpful, e.g. for ints
# For strings of np.object, this should not be necessary
one_hot_int_cols = df.dtypes[df.dtypes == np.int].index.tolist()
for col in one_hot_int_cols:
    df[col] = pd.Categorical(df[col])

# One-hot encoding of features: Dummy variables with pandas
# Use drop_first=True to remove the first category and avoid multi-colinearity
col_dummies = ['var1', 'var2']
try:
    for col in col_dummies:
        df = pd.concat([df.drop(col, axis=1),
        				pd.get_dummies(df[col], prefix=col, prefix_sep='_',
        					drop_first=True, dummy_na=False)],
                        axis=1)
except KeyError as err:
    print("Columns already dummified!")

# Train/Test split
# If classification, use a stratified version to keep class ratios!
y = df['price']
X = df.drop('price',axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, # predictive variables
    y, # target
    test_size=0.1, # portion of dataset to allocate to test set
    random_state=42 # we are setting the seed here, ALWAYS DO IT!
    # stratify=y # if we want to keep class ratios in splits
) # We can also use the stratify argument: stratify = X[variable]

# Stratified splits for classification 
# A more advanced way of creating splits with constant class ratios
# StratifiedShuffleSplit allows to split the dataset into the desired numbers of train-test subsets
# while still maintaining the ratio of the predicted classes in the original/complete dataset
from sklearn.model_selection import StratifiedShuffleSplit
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, # 1 split: 1 train-test
                                          test_size=0.3, 
                                          random_state=42)
# Get the split indexes
train_idx, test_idx = next(strat_shuf_split.split(X=df[feature_cols], y=df[target]))
# Create the dataframes using the obtained split indices
X_train = df.loc[train_idx, feature_cols]
y_train = df.loc[train_idx, target]
X_test  = df.loc[test_idx, feature_cols]
y_test  = df.loc[test_idx, target]
# Always check that the ratios are OK, ie., very similar for all subsets
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

# Encoding of categorical variables if target is categorical/binary
# Loop over all categorical columns
# Create new variable which contains the target ratio
# associated with each category/level
categorical_cols_encoded = []
for col in categorical_cols:
	# Category levels for each col: target mean for each level
    col_groups = df.groupby(col).mean()['target']
    # Value column: target mean of the corresponding level
    col_values = []
    for val in data[col]:
        col_values.append(col_groups.loc[val])

    col_encoded_name = col + '_target'
    categorical_cols_encoded.append(col_encoded_name)
    df[col_encoded_name] = col_values

# Fit the scaler to the train set only!
# Try different scalers: StandardScaler(), RobustScaler(), etc.
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

# Always save the scaler + any transformer object + parameters!
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

# Define a Lasso model with a set of possible alphas, i.e., regularization strengths
# Get with Cross-Validation the optimum alpha
alphas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
model = LassoCV(alphas=alphas,
                max_iter=5e4,
                cv=4)
# Train/Fit auxiliary Lasso model
model.fit(X_train, y_train)
# Use the optimum alpha obtained with Cross-Validation
# Set and save the seed for reproducibility
sel_ = SelectFromModel(Lasso(alpha=model.alpha_, random_state=42))
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

## T-Test of independent samples: 2 means

# Charges of smokers are larger for smokers
smoker = df.loc[df.smoker=="yes"]
smoker_char = smoker.charges
nonsmoker = df.loc[df.smoker=="no"]
nonsmoker_char = nonsmoker.charges
sns.boxplot(x=df.charges,y=df.smoker,data=data).set(title="Smoker vs Charges")

# Test: T-test, one-sided
# H0: mean_smoker <= mean_nonsmoker
# Ha: mean_smoker > mean_nonsmoker
alpha = 0.05
t_val, p_value = stats.ttest_ind(smoker_char, nonsmoker_char)
p_value_onetail = p_value/2 # p_value -> 0 (reject H0)

## One-way ANOVA: >2 means

# BMI of women with no children, one child, and two children
female = df[df.gender == 'female']
female_children = female.loc[female['children']<=2]
sns.boxplot(x="children", y="bmi", data=female_children)

# Test: ANOVA
# Build OLS model with formula and compute ANOVA table
# C(): treat as categorical even though it is an integer
formula = 'bmi ~ C(children)'
model = ols(formula, female_children).fit()
aov_table = anova_lm(model)
aov_table # P(>F) = 0.715858 (cannot reject H0 -> there is no difference between means)

### Chi-square test: >2 proportions & contingency tables

# Is the proportion of smokers is significantly different across the different regions?
# Create contingency table
contingency = pd.crosstab(df.region, df.smoker)
contingency

# Plot as bar chart
contingency.plot(kind='bar')

# Test: Chi-sqare
# H0: Smokers proportions are not significantly different across the different regions. 
# Ha: Smokers proportions are different across the different regions.
# p_val = 0.06171954839170541 (cannot reject H0)
chi2, p_val, dof, exp_freq = chi2_contingency(contingency, correction = False)
print('chi-square statistic: {} , p_value: {} , degree of freedom: {} ,expected frequencies: {} '.format(chi2, p_val, dof, exp_freq))

# Power analysis:
# The following four variables are related (i.e., fixing 3, we predict the 4th):
# - Effect size: Cohen's d between groups : d = (mean_1 - mean_2) / unpooled_std
# - Sample size (n)
# - Significance level (alpha)
# - Statistical power (1-beta)
# Estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
# parameters for power analysis
effect = 0.8 # Cohen's d
alpha = 0.05
power = 0.8 # 1 - beta
# perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)

##### -- 
##### -- Data Modelling
##### -- 

### --- Summary of the most important models

# Choose and instantiate the model
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
model = LinearRegression()
# Regularized: we can also do cross-validation, see below: LassoCV, RidgeCV, etc.
model = Lasso(alpha=0.001, random_state=0) # alpha: regularization strength
model = Ridge(alpha=0.001, random_state=0) 
model.fit(X_train, y_train)
model.coef_
# ---
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
model = LogisticRegression()
# Although logistic regression is binary, it is generalized in Scikit-Learn
# and it can take several classes; the target needs to be LabelEncoded.
# If we use the solver 'liblinear', the 'one-vs-rest' approach is used,
# with other solvers, a multinomial approach can be used.
# Also, we can use the cross-validation version to detect the optimum C value
model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear')
model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
model = LogisticRegression(random_state=101, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter = 1000)
# ---
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
model = KNeighborsClassifier(n_neighbors=3) # test in a loop best k = n_neighbors
# n_neighbors should be multiple of the number of classes + 1
# Use the elbow method to get best K: vary K in a for loop and choose model with best metric
# KNeighborsRegressor computes the weighted target value of the K nearest neighbors
# ---
# SGDClassifier: Linear classifiers (SVM, logistic regression, etc.) with SGD training;
# depending on the loss, a model is used - default loss: hinge -> SVM
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
# LinearSVC is the linear SVM model, to be used with small simple datasets
# SVC is the non-linear SVM model that uses kernels, e.g., the Gaussian kernel (RBF)
# SVC can be used with small to medium datasets
# We can define:
# - gamma: multiplier factor of the kernel
# - C: 1/lambda for regularization
# - large C and gamma are related to more complex and curvy models 
model = SVC(kernel='rbf', gamma=1.0, C=10.0)
# If we have a large dataset, we should use a kernel approximation
# to transform the dataset and the linear SVM: LinearSVC or SGDClassifier
from sklearn.kernel_approximation import Nystroem
# n_components: number of landmarks to compute kernels; these are the new features
NystroemSVC = Nystroem(kernel="rbf", gamma=1.0, n_components=100)
X_train = NystroemSVC.fit_transform(X_train)
X_test = NystroemSVC.transform(X_test)
sgd = SGDClassifier()  # loss="hinge" by default, so a SVM is used
linSVC = LinearSVC()
linSVC.fit(X_train, y_train)
sgd.fit(X_train, y_train)
y_pred_linsvc = linSVC.predict(X_test)
y_pred_sgd = sgd.predict(X_test)
# ---
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# Without any constraints, our decision tree will overfit the training dataset.
# We can use this overfit model to identify the maximum depth and feature numbers
# for a grid search later on.
# Regularizing parameters:
# - criterion: we can select different information curves to decide whether to further split: 'gini', 'entropy'
# - max_features: maximum number of features to look at when we split
# - max_depth: maximum allowed depth
# - min_samples_leaf: minimum samples necessary to be a leaf (default: 1)
model = DecisionTreeClassifier(random_state=42)
model = model.fit(X_train, y_train)
# We define the search array maximum values to be the values of the overfit tree
param_grid = {'max_depth':range(1, model.tree_.max_depth+1, 2),
              'max_features': range(1, len(model.feature_importances_)+1)}
# Grid search with cross validation to determine the optimum hyperparameters
gt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1)
gt = gt.fit(X_train, y_train)
# Get best estimator: the tree and its parameters
gt.best_estimator_.tree_.node_count, GR.best_estimator_.tree_.max_depth
# Get feature importances
gt.best_estimator_.feature_importances_
## Plot tree
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
# Create an output destination for the file
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# View the tree image
# The nodes are color coded: each color belongs to a class,
# the darker the color the more pure the contained data-points.
filename = 'wine_tree.png'
graph.write_png(filename)
Image(filename=filename) 
# ---
# Random forests train n_estimators trees in subsets of the dataset
# constraining the number features used in each independent parallelized tree.
# They cannot overfit, but their performance plateaus from a given n_estimators on.
# We can detect the critical n_estimators trying different values in a for-loop.
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(oob_score=True, # compute Out-of-bag score: score out of sub-sample
                            random_state=42, 
                            warm_start=True, # Re-use the previous result to fit new tree
                            n_jobs=-1) # use all CPUs
oob_list = list()
# Iterate through all of the possibilities for number of trees
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    # Set number of trees now
    RF.set_params(n_estimators=n_trees)
    RF.fit(X_train, y_train)
    # Get the oob error: score on points out of samples
    oob_error = 1 - RF.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))
# Plot how the out-of-bag error changes with the number of trees
rf_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')
rf_oob_df.plot(legend=False, marker='o', figsize=(14, 7), linewidth=5)
# Set the optimum number of trees and extract feature importances.
# Apparently, we don't need to train the model again.
# I understand that's because the last trained model had 400 estimators;
# so I guess we take 100 from those 400?
model = RF.set_params(n_estimators=100)
feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_imp.plot(kind='bar', figsize=(16, 6))
# ---
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
model.feature_importances_
# ---
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# Gradient Boosting improves weak learner trees successively penalizing residuals,
# thus, it risks of overfitting; solution: we apply grid search.
# Grid Search: hyperparameters
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [400], #tree_list,
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [1.0, 0.5],
              'max_features': [1, 2, 3, 4]}
GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42,
                                                 warm_start=True), 
                      param_grid=param_grid, 
                      scoring='accuracy',
                      n_jobs=-1)
# Do the grid search fittings
GV_GBC = GV_GBC.fit(X_train, y_train)
# The best model
GV_GBC.best_estimator_
# ---
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
# Voting and staking combine different models; 
# the final answer is the result of a voting or an estimation out of model outputs.
# We need to define the parallel models or estimators.
# Here, estimators are defined with default hyperparams;
# we should better use either specific params or perform grid search.
estimators = [('SVM',SVC(random_state=42)),
              ('knn',KNeighborsClassifier()),
              ('dt',DecisionTreeClassifier())]
# Voting: 'hard' if we want majority class, 'soft' if probabilities are averaged
VC = VotingClassifier(estimators=estimators, voting='soft')
# Fit and predict in the case of voting
VC = VC.fit(X_train, y_train)
y_pred = VC.predict(X_test)
# Stacking: a final estimator is required
SC = StackingClassifier(estimators=estimators, final_estimator= LogisticRegression())
# Grid search: hyperparameters
# Recall to use double _ for models within model
param_grid = {'dt__max_depth': [n for n in range(10)],
              'dt__random_state':[0],
              'SVM__C':[0.01,0.1,1],
              'SVM__kernel':['linear', 'poly', 'rbf'],
              'knn__n_neighbors':[1,4,8,9]}
search = GridSearchCV(estimator=SC, param_grid=param_grid,scoring='accuracy')
search.fit(X_train, y_train)
search.best_score_ # 1, be aware of the overfitting!
search.best_params_
# ---
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# ---
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=2,covariance_type='diag')
# ---
from sklearn.mixture import BayesianGaussianMixture
model = BayesianGaussianMixture(n_components=2,covariance_type='diag')
# ---
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
# ---
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(X)
model.cluster_centers_
model.labels_
# ---
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X)
X_reduced = model.transform(X)
model.components_

# Fit / Train
# Sometimes, data frames need to be converted to 1D arrays
model.fit(X_train, y_train)
model.fit(X_train, np.ravel(y_train))

# Predict 
pred_train = model.predict(X_train)
pred_test = model.predict(X_test) # if classification, LabelEncoded target integer
# If classification, we can get probabilities of classes, each class a probability value (column); select max
prob_test = model.predict_proba(X_test).max(axis=1)

### --- Evaluation and Interpretation: General

# If we want to convert text tables as figure (for saving)
plt.figure(figsize=(5, 5))
plt.text(0.01, 1.25, str('Table Title'), {'fontsize': 10}, fontproperties = 'monospace')
plt.text(0.01, 0.05, str(classification_report(y_train, y_pred)), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')

### --- Classification: Evaluation and Interpretation

# Precision, recall, f-score, support:
# For multiple classes, there is one value for each class
# However, we can compute the weighted average to get a global value with average='weighted'.
# Then, support doesn't make sense.
# Without the average parameter, we get arrays of six values for each metric,
# one item in each array for each class.
precision, recall, fscore, _ = classification_score(y_test, y_pred[lab], average='weighted')
# Accuracy is for the complete dataset (ie., all classes).
accuracy = accuracy_score(y_test, y_pred[lab])

# Confusion matrix; classification report: accuracy, precision, recall, F1
print(confusion_matrix(y_test,pred_test))
print(classification_report(y_test,pred_test))
sns.heatmap(confusion_matrix(y_test,pred_test), annot=True);

# ROC-AUC scores can be calculated by binarizing the data
# label_binarize performs a one-hot encoding,
# so from an integer class we get an array of one 1 and the rest 0s.
# This is necessary for computing the ROC curve, since the target needs to be binary!
# Again, to get a single ROC-AUC from the 6 classes, we pass average='weighted'
auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
          label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
          average='weighted')
model_roc_plot = plot_roc_curve(model, X_test, y_test, name="Logistic Regression") # ROC curve plotted and AUC computed
# An alternative is the following, but we need to pass y_prob = model.predict_proba(X_test)
# But we need to plot manually
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1]) # select the class from which we need the probabilities
plt.plot(fpr, tpr, linewidth=5)
plt.plot([0, 1], [0, 1], ls='--', color='black', lw=.3)
plt.set(xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        xlim=[-.01, 1.01], ylim=[-.01, 1.01],
        title='ROC curve')
plt.grid(True)

# Precision-Recall Curve: Similar to ROC, but better suited for unbalanced datasets
# We need to plot manually
precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
plt.plot(recall, precision, linewidth=5)

### --- Classification: Decision Boundary Plotting

def plot_decision_boundary(estimator, X, y, label_0, label_1):
    estimator.fit(X, y)
    X_color = X.sample(300) # We take only 300 points, because otherwise we have too many points
    y_color = y.loc[X_color.index] # We take the associated y values of the sampled X points
    y_color = y_color.map(lambda r: 'red' if r == 1 else 'yellow')
    x_axis, y_axis = np.arange(0, 1, .005), np.arange(0, 1, .005) # very fine cells
    xx, yy = np.meshgrid(x_axis, y_axis) # cells created
    xx_ravel = xx.ravel()
    yy_ravel = yy.ravel()
    X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T
    y_grid_predictions = estimator.predict(X_grid) # for each cell, predict values
    y_grid_predictions = y_grid_predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3) # plot regions and boundary
    ax.scatter(X_color.iloc[:, 0], X_color.iloc[:, 1], color=y_color, alpha=1) # 300 sampled data-points
    ax.set(
        xlabel=label_0,
        ylabel=label_1,
        title=str(estimator))

# Here, different decision boundaries
# of an SVC with varied C and gamma values are plotted.
# An example figure is above, prior to the code.

# Insights:
# - Higher values of gamma lead to LESS regularization, i.e., more curvy and complex models
# - Higher values of C lead to LESS regularization, i.e., more curvy and complex models 

from sklearn.svm import SVC # Support Vector Machine classifier

gammas = [.5, 1, 2, 10]
for gamma in gammas:
    SVC_Gaussian = SVC(kernel='rbf', gamma=gamma)
    plot_decision_boundary(SVC_Gaussian, X, y, label_0='feature_1', label_1='feature_2')

### --- Regression: Evaluation and Interpretation

# Evaluate; consider whether the targe was transformed or not
# If Regression: MSE, R2
print('MAE: ', mean_absolute_error(np.exp(y_test), np.exp(pred_test)))
print('RMSE: ', mean_squared_error(np.exp(y_test), np.exp(pred_test), squared=False))
print('R2', r2_score(np.exp(y_test), np.exp(pred_test)))

# If Regression: Plot True y vs. Predicted y
plt.figure(figsize=(8,8))
y_true = np.exp(y_test)
y_pred = np.exp(model.predict(X_test))
plt.scatter(y_true, y_pred,color='b',alpha=0.5)
plt.plot(np.array([0,np.max(y_true)],dtype='object'),np.array([0,np.max(y_true)],dtype='object'),'r-')
plt.xlabel('True Target')
plt.ylabel('Predicted Target')
plt.axis('equal')

# If Regression: check residuals are normally distributed (QQ plot)
y_true = np.exp(y_test)
y_pred = np.exp(model.predict(X_test))
error = y_true - y_pred
stats.probplot(error, dist="norm", plot=plt);

# If Regression: Get & plot model coefficients / feature importances
# However, note that significance p-values are missing here: in addition to large coefficients,
# we should check their p-value; statsmodels can do that
importance = pd.DataFrame(model.coef_.ravel()) # LinearRegression, Lasso, Ridge
importance = pd.DataFrame(model.feature_importances_.ravel()) # RandomForestRegressor
importance.index = df.columns
importance.columns = ['coef']
importance['plus'] = importance['coef'].apply(lambda col: 1 if col > 0 else 0)
importance['coef'] = np.abs(importance['coef'])
# Plot top k coefficients
top_features = 30
plt.figure(figsize=(6,10))
importance.sort_values(by='coef',ascending=True,inplace=True)
color_list = ['b' if el > 0 else 'r' for el in [importance['plus'].iloc[i] for i in range(importance['plus'].shape[0])]]
importance['coef'][-top_features:].plot(kind='barh',color=color_list[-top_features:])
plt.xlabel('Coefficient Value of Features')

### --- Evaluation and Interpretation of Black Box Models: Permutation Feature Importance

# Feature importances are obtained by shuffling the values in each column and comparing the prediction error
# Typical black box models: 
# - NNs
# - SVM with non-linear kernels
# - Random Forests (although they have feature_importances_)
# - Gradient boosted trees

# Use permutation_importance to calculate permutation feature importances.
# Note we have these parameters, too:
# - n_repeats: how many times each feature is permuted
# - sample_weight: weight assigned to each sample/data-point
# The output is of the size: n_features x n_repeats
feature_importances = permutation_importance(estimator=black_box_model,
                                             X = X_train,
                                             y = y_train,
                                             n_repeats=5,
                                             random_state=123,
                                             n_jobs=2)
feature_importances.importances # (11, 5) array: features x n_repeats
# Plot
def visualize_feature_importance(importance_array):
    # Sort the array based on mean value
    sorted_idx = importance_array.importances_mean.argsort()
    # Visualize the feature importances using boxplot
    fig, ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    fig.tight_layout()
    ax.boxplot(importance_array.importances[sorted_idx].T,
               vert=False, labels=X_train.columns[sorted_idx])
    ax.set_title("Permutation Importances (train set)")
    plt.show()
# A ranked box plot is shown, with a box for each feature
# Note that we used n_repeats=5;
# we can increase that number to have more realistic box plots
visualize_feature_importance(feature_importances)

### --- CROSS-VALIDATION: Hyperparameter Tuning

# We can loop across different parameter values and find the set
# that yields the optimum metric.
# Instead of doing it manually, we can do it with GridSearchCV

from sklearn.model_selection import GridSearchCV, KFold

# Set estimator Pipeline: all necessary feature engineering steps
# with parameters to tune + scaling + model
estimator = Pipeline([
    ("polynomial_features", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("ridge_regression", Ridge())])

# Definition of k=3 mutually exclusive validation splits
# Alternatives if class ratios need to be kept: StratifiedKFolds
kf = KFold(shuffle=True, random_state=72018, n_splits=3)

# We compose a dictionary with parameter values to test or to look up
# If the estimator is a model: {'paramater':[value_array]}
# If the estimator is a Pipeline: {'object_name__parameter':[value_array]}; note the double '_'!
params = {
    'polynomial_features__degree': [1, 2, 3],
    'ridge_regression__alpha': np.geomspace(1e-3, 20, 30) # locagithmic jumps
}

# Instantiate a grid search for cross-validation
grid = GridSearchCV(estimator, params, cv=kf)

# Find the optimal parameters
# Basically, the estimator is fit going through all parameter combinations:
# 3 degrees x 30 alphas = 90 combinations
grid.fit(X, y) # X_train, y_train

# Get best values: cross-validation score and parameters associated with it
grid.best_score_, grid.best_params_

# The best parameter set is taken and the estimator used to predict
# Notice that "grid" is a fit object!
# We can use grid.predict(X_test) to get brand new predictions!
y_predict = grid.predict(X) # X_test

# This includes both in-sample and out-of-sample
r2_score(y, y_predict) # y_test, y_predict

# We can access any Pipeline object of the best estimator
# and their attributes & methods!
# Here, the model coefficients
grid.best_estimator_.named_steps['ridge_regression'].coef_

# Get the model statistics/properties of each parameter combination
pd.DataFrame(grid.cv_results_)

## Cross-Validation Alternative: Use models with built-in cross-validation

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Regularization parameters to test
alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]
l1_ratios = np.linspace(0.1, 0.9, 9)
# Model definition
model = RidgeCV(alphas=alphas, cv=4, random_state=0)
model = LassoCV(alphas=alphas,
                max_iter=5e4,
                cv=3,
                random_state=0)
model = ElasticNetCV(alphas=alphas, 
                     l1_ratio=l1_ratios,
                     max_iter=1e4,
                     random_state=0)
# Train/Fit
model.fit(X_train, y_train)
# Extract values
model.alpha_
model.coef_
np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

### --- Unbalanced Datasets

# 1. Perform stratified train/test split
# 2. Get metrics with original dataset: precision, recall, f1; select the most appropriate one
#       - Precision measures how bad the Type I error is
#       - Recall measures how bad the Type II error is
# 3. Try techniques:
#       - Weights: inverse of class ratios passed as dictionary in model instantiation
#       - Resampling: oversampling (SMOTE) or undersampling
# 4. Compute metrics again and take best technique for our case

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state = rs)

# Class weights can be passed in a dictionary
# choosing values which are inverse to the frequency of the class.
# For instance, if class 0 : class 1 has a ratio of 10:1, we could define:
class_weight = {}
class_weight[0] = 0.1 # 10:1
class_weight[1] = 0.9 # 1:10
model = RandomForestClassifier(random_state=rs, class_weight=class_weight)

# However, we can also treat class weights as hyperparameter to be tuned
params_grid = {
  'max_depth': [5, 10, 15, 20],
  'n_estimators': [25, 50, 100],
  'min_samples_split': [2, 5],
  'class_weight': [{0:0.1, 1:0.9}, {0:0.2, 1:0.8}, {0:0.3, 1:0.7}]
}
model = RandomForestClassifier(random_state=rs)
grid_search = GridSearchCV(estimator = model, 
                       param_grid = params_grid, 
                       scoring='f1',
                       cv = 5, verbose = 1)

# Metric computation
# Parameters (case: binary classification):
# - beta: the strength of recall versus precision in the F-score; default 1.0
# - pos_label: the class to report if average='binary' and the data is binary; default 1
precision, recall, fbeta, support = classification_score(y_test, preds, beta=1, pos_label=1, average='binary')

# Resampling: under- and oversampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

def resample(X_train, y_train):
    # Oversampling: minority class(es) synthetically multiplied
    # SMOTE oversampler: new data-points between a minority point and its nearest neighbors
    smote_sampler = SMOTE(random_state = 123)
    # Undersampling: number of points of majority class decreased
    # Random undersampler: removed points randomly selected
    under_sampler = RandomUnderSampler(random_state=123)
    # Resampled datasets
    X_smo, y_smo = smote_sampler.fit_resample(X_train, y_train)
    X_under, y_under = under_sampler.fit_resample(X_train, y_train)
    return X_smo, y_smo, X_under, y_under

X_smo, y_smo, X_under, y_under = resample(X_train, y_train)
# Now, fit new model and compute its metrics