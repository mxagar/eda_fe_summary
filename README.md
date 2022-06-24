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

## Data Cleaning

- Always do general checks: `head()`, `info()`, `describe()`, `shape`.
- Always get lists of column types: `select_dtypes()`.
	- Categorical columns, `object`: `unique()`, `value_counts()`.
	- Numerical columns, `int`, `float`: `describe()`.
- Detect and remove duplicates: `duplicated()`, `drop_duplicates()`.
- Correct inconsistent text/typos in labels, use clear names: `replace()`, `map()`.
- Detect and fix missing data: 
	- `dropna()` rows iff several fields missing or missing field is key (e.g., target).
	- `drop()` columns if many (> 20-30%) values are missing. 
	- Impute the missing field/column with `fillna()`: `mean()`, `meadian()`, `mode()`.
	- Mask the data: create a category for missing values, in case it leads to insights.
	- More advanced:
		- Predict values with a model.
		- Use k-NN to impute values of similar data-points.
- Detect and handle outliers:
	- Histogram plots: `sns.histplot()`.
	- Box plots: `sns.boxplot()`.
	- Scatterplots: `plt.scatter()`, `plt.plot()`.
	- Residual plots: differences between the real/actual target values and the model predictions.
	- Compute `stats.zscore()` to check how many std. deviations from the mean; often Z > 3 is considered an outlier.
	- IQR calculation: use `np.percentile()`.
	- Drop outliers? Only iff we think they are not representative.
	- Do transformations fix the `skew()`? `np.log()`, `np.sqrt()`, `stats.boxcox()`.
		- Usually a absolute skewness larger than 0.75 requires a transformation (feature engineering).

## Exploratory Data Analysis



## Feature Engineering

- Transformations: if variables have a `skew()` larger than 0.75.
	- Target: usually the logarithm is applied: `df[col] = df[col].apply(np.log1p)`.
		- That makes undoing the transformation very easy: `np.exp(pred)`.
		- If power transformations used (e.g., `boxcox`, `yeojohnson`), we need to save the params/transformer and make sure we know how to invert the transformation!
	- Predictor(independent) variables: `scipy` or `sklearn` can be used for power transformations (e.g., `boxcox`, `yeojohnson`). 
- Extract / create new features, more descriptive
	- Multiply different features if we suspect there might be an interaction
	- Divide different features, if the division has a meaning.
	- Create categorical data from continuous it that has a meaning.
	- Try polynomial features: `PolynomialFeatures()`.
	- Create deviation factors from the mean of a numeric variable in groups or categories of another categorical variable. 

- Replace categorical levels with few counts with `'other'`

- Feature encoding
	- One-hot endoding / dummy variables: `get_dummies()`.
- Feature scaling
	- `StandardScaler()`: subtract the mean and divide by the standard deviation; features are converted to standard normal viariables.
	- `MinMaxScaler()`: a mapping with which the minimum value becomes 0, max becomes 1. This is senstive to outliers!
	- `RobustScaler()`: IQR range is mapped to `[0,1]`, i.e., percentiles `25%, 75%`; thus, the scaled values go out from the `[0,1]` range.

## Feature Selection

- `SelectFromModel(Lasso())`
- Use pairplots to check multi-colineearity; correlated features are not good.

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
