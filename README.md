# Data Processing: A Practical Guide

The steps in the data science pipeline that need to be carried out to answer business questions are:

1. Data Understanding & Formulation of the Questions
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Feature Selection
6. Data Modelling

The file [data_processing.py](data_processing.py) compiles the most important tools for the steps 2-5 I use, following the [80/20 Pareto principle](https://en.wikipedia.org/wiki/Pareto_principle). Additionally, in the following, some practical guidelines are summarized very schematically.

Note that this guide assumes familiarity with `python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn` and `scipy`, among others. Additionally, I presume you are acquainted machine learning and data science concepts.

For more information on the motivation of the guide, see my [blog post](https://mikelsagardia.io/blog/data-processing-guide.html).

### Table of Contents

- [General](#general)
- [Data Cleaning](#Data-Cleaning)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Feature Selection](#Feature-Selection)
- [Hypothesis Tests](#Hypothesis-Tests)
- [Tips for Production](#Tips-for-Production)
- [Relevant Links](#Relevant-Links)
- [Authorship](#Authorship)


## General

- Watch at the returned types
	- Is it a collection or contained? Convert it to a `list()`.
	- Is it an array/tuple with one item: access it with `[0]`.
- Data frames and series can be sorted: `sort_values(by, ascending=False)`.
- Recall we can use handy python data structures:
	- `set()`
	- `Counter()`
- Use `np.log1p()` in case you have `x=0`; `log1p(x) = log(x+1)`
- Use `df.apply()`!
- Make a copy of the dataset if we drop or change variables: `data = df.copy()`.
- Categorical variables must be enconded as quantitative variables.
- Seaborn plots get `plt.figure(figsize=(10,10))` beforehand; pandas plots get `figsize` are argument.
- `plt.show()` only on scripts!
- Use a seed whenever there is a random number generation to ensure reproducibility!
- Pandas slicing:
	- `df[]` should access only to column names/labels: `df['col_name']`.
	- `df.iloc[]` can access only to row & column index numbers + booleans: `df.iloc[0,'col_name']`, `df.iloc[:,'col_name']`.
	- `df.loc[]` can access only to row & column labels/names + booleans: `df.loc['a','col_name']`, `df.loc[:,'col_name']`.
	- `df[]` can be used for changing entire column values, but `df.loc[]` or `df.iloc[]` should be used for changing sliced row values.

## Data Cleaning

- Always do general checks: `head()`, `info()`, `describe()`, `shape`.
- Always get lists of column types: `select_dtypes()`.
	- Categorical columns, `object`: `unique()`, `value_counts()`. Can be subdivided in:
		- `str`: text or category level.
		- `datetime`: encode with `to_datetime()`.
	- Numerical columns, `int`, `float`: `describe()`.
- Detect and remove duplicates: `duplicated()`, `drop_duplicates()`.
- Correct inconsistent text/typos in labels, use clear names: `replace()`, `map()`.
- Detect and fix missing data: 
	- Plot the missing amounts: `df.isnull().sum().sort_values(ascending=False)[:10].plot(kind='bar')`.
	- `dropna()` rows if several fields missing or missing field is key (e.g., target).
	- `drop()` columns if many (> 20-30%) values are missing. 
	- Impute the missing field/column with `fillna()` if few (< 10%) rows missing: `mean()`, `meadian()`, `mode()`.
	- Mask the data: create a category for missing values, in case it leads to insights.
	- More advanced:
		- Predict values with a model.
		- Use k-NN to impute values of similar data-points.
- Detect and handle outliers:
	- Keep in mind the empricial rule of 68-95-99.7 (1-2-3 std. dev.).
	- Compute `stats.zscore()` to check how many std. deviations from the mean; often Z > 3 is considered an outlier.
	- Histogram plots: `sns.histplot()`.
	- Box plots: `sns.boxplot()`.
	- Scatterplots: `plt.scatter()`, `plt.plot()`.
	- Residual plots: differences between the real/actual target values and the model predictions.
	- IQR calculation: use `np.percentile()`.
	- Drop outliers? Only iff we think they are not representative.
	- Do transformations fix the `skew()`? `np.log()`, `np.sqrt()`, `stats.boxcox()`.
		- Usually a absolute skewness larger than 0.75 requires a transformation (feature engineering).

## Exploratory Data Analysis

- Recall we have 3 main ways of plotting:
	- Matplotlib: default: `plt.hist()`.
	- Seaborn: nicer, higher interface: `sns.histplot()`.
	- Pandas built-in: practical: `df.plot(kind='hist')`, `df.hist()`, `df.plot.hist()`.
- We need to plot / observe **every** feature or variable:
	- Create automatic lists of variable types: `select_dtypes()`. Usual types:
		- Numerical: `int`, `float`.
		- Strings: `object`. Can contain:
			- `strings`: text or category level.
			- `dates`: encode with `to_datetime()`.
	- Loop each type list and apply the plots/tools we require.
- Quantitative/numerical variables: can be uni/multi-variate; most common EDA tools:
	- Histograms: `sns.histplot()`, `plt.hist()`, `df.hist()` - look at: shape, center, spread, outliers.
	- Numerical summaries: `describe()`.
	- Boxplots: `sns.boxplot()` - look at outliers.
		- Also, combine these two: `sns.catplot()`, `sns.stripplot()`.
	- Scatteplots: `sns.scatterplot()`, `plt.scatter()`, `sns.regplot()`, `sns.lmplot()` - look at: linear/quadratic relationship, positive/negative, strength: weak/moderate/strong.
		- Beware of the [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox).
	- Correlations: `df.corr()`,  `stats.pearsonr()`; see below.
		- Beware of the [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox).
- Categorical variables: ordinal (groups have ranking) / cardinal (no order in groups); most common EDA tools:
	- Count values: `unique()`, `value_counts().plot(kind='bar')`.
	- Frequency tables; see below.
	- Bar charts: `sns.barplot()`, `plt.bar()`, `plt.barh()`; use `sort_values()`.
	- Count plots: `sns.countplot()`.
- If a continuous variable has different behaviors in different ranges, consider stratifying it with `pd.cut()`. Example: age -> age groups.
- Frequency tables: stratify if necessary, group by categorical levels, count and compute frequencies.
- Correlations:
	- Heatmap for all numerical variables: `sns.heatmap(df.corr())`.
	- Bar chart for correlations wrt. target: `df.corr()['target'].sort_values(ascending=True).plot(kind='bar')`.
	- Pair correlations: `stats.pearsonr(df['x'],df['y'])`.
	- Check if there is multicolinearity: it's not good.
	- Beware of confounding: two correlated variables can be affected/related by something different.
- If you want to try different or plots:
	- Historgrams: add density, `kde=True` in `sns.histplot()`.
	- Boxplots -> try `sns.swarmplot()`: boxes are replaced by point swarms.
	- Boxplots -> try `sns.violinplot()`: box width changed.
	- Scatterplots -> try `sns.lmplot()` or `sns.reglot()`: linear regression is added to the scatterplot.
	- Scatterplots -> try `sns.jointplot()`: density distribution isolines of a pair of quantitative variables.
- We can get a handle of any plot and set properties to is later: `bp = sns.boxplot(); bp.set_xlabel()`.
- Larger group plots:
	- `sns.pairplot()`: scatterplots/histograms of quantitative varaibles in a matrix; select variables if many.
	- `sns.FacetGrid()`: create a grid according to classes and map plot types and variables.


## Feature Engineering

- Always make a copy of the dataset if we change it: `df.copy()`.
- Transformations: if variables have a `skew()` larger than 0.75.
	- Target: usually the logarithm is applied: `df[col] = df[col].apply(np.log1p)`.
		- That makes undoing the transformation very easy: `np.exp(pred)`.
		- If power transformations used (e.g., `boxcox`, `yeojohnson`), we need to save the params/transformer and make sure we know how to invert the transformation!
	- Predictor /independent variables: `scipy` or `sklearn` can be used for power transformations (e.g., `boxcox`, `yeojohnson`). 
- Extract / create new features, more descriptive
	- Multiply different features if we suspect there might be an interaction
	- Divide different features, if the division has a meaning.
	- Create categorical data from continuous it that has a meaning, e.g., daytime.
	- Try polynomial features: `PolynomialFeatures()`.
	- Create deviation factors from the mean of a numeric variable in groups or categories of another categorical variable. 
- Replace categorical levels with few counts with `'other'`.
- Feature encoding
	- One-hot encoding / dummy variables: `get_dummies()`.
- Train/test split: perform it before scaling the variables: `train_test_split()`.
- Feature scaling: apply it if data-point distances are used in the model; fit the scaler only with the train split!
	- `StandardScaler()`: subtract the mean and divide by the standard deviation; features are converted to standard normal viariables.
	- `MinMaxScaler()`: a mapping with which the minimum value becomes 0, max becomes 1. This is senstive to outliers!
	- `RobustScaler()`: IQR range is mapped to `[0,1]`, i.e., percentiles `25%, 75%`; thus, the scaled values go out from the `[0,1]` range.

## Feature Selection

- We can measure sparsity of information with `PCA()`; if less variables explain most of the variance, we could drop some.
- Select variables with L1 regularized regression (lasso): `SelectFromModel(Lasso())`. L1 regularization forces coefficients of less important variables to become 0; thus, we can remove them.
- Use pairplots to check multi-colinearity; correlated features are not good.

## Data Modelling



## Hypothesis Tests



## Tips for Production



## Relevant Links

- [machine_learning_ibm](https://github.com/mxagar/machine_learning_ibm)
- [statistics_with_python_coursera](https://github.com/mxagar/statistics_with_python_coursera)
- [deploying-machine-learning-models](https://github.com/mxagar/deploying-machine-learning-models)
- [airbnb_data_analysis](https://github.com/mxagar/airbnb_data_analysis)

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You can freely use and forward this repository if you find it useful. In that case, I'd be happpy if you link it to me :blush:.
