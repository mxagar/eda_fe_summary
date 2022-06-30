# Data Processing: A Practical Guide

The steps in the data science pipeline that need to be carried out to answer business questions are:

1. Data Understanding & Formulation of the Questions
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Feature Selection
6. Data Modelling

The file [data_processing.py](data_processing.py) compiles the most important tools I use for the steps 2-5, following the [80/20 Pareto principle](https://en.wikipedia.org/wiki/Pareto_principle). Additionally, in the following, some practical guidelines are summarized very schematically.

Note that this guide assumes familiarity with `python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn` and `scipy`, among others. Additionally, I presume you are acquainted with machine learning and data science concepts.

For more information on the motivation of the guide, see my [blog post](https://mikelsagardia.io/blog/data-processing-guide.html).

### Table of Contents

- [General](#general)
- [Data Cleaning](#Data-Cleaning)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Feature Engineering](#Feature-Engineering)
	- [Scikit-Learn Transformers](#Scikit-Learn-Transformers)
	- [Creation of Transformer Classes](#Creation-of-Transformer-Classes)
		- Manual definition
		- Classes derived from `sklearn`
- [Feature Selection](#Feature-Selection)
- [Hypothesis Tests](#Hypothesis-Tests)
- [Data Modelling](#Data-Modelling)
- [Tips for Production](#Tips-for-Production)
	- [Pipelines](#Pipelines)
- [Relevant Links](#Relevant-Links)
- [Authorship](#Authorship)

## General

- Watch at the returned types:
	- If it is a collection or a container, convert it to a `list()`.
	- If it is an array/tuple with one item, access it with `[0]`.
- Data frames and series can be sorted: `sort_values(by, ascending=False)`.
- Recall we can use handy python data structures:
	- `set()`: sets of unique elements.
	- `Counter()`: dict subclass for counting hashable objects.
- Use `np.log1p()` in case you have `x=0`; `log1p(x) = log(x+1)`.
- Use `df.apply()` extensively!
- Make a copy of the dataset if you drop or change variables: `data = df.copy()`.
- All categorical variables must be enconded as quantitative variables somehow.
- Seaborn plots get `plt.figure(figsize=(10,10))` beforehand; pandas plots get `figsize` as argument.
- `plt.show()` only in scripts!
- Use a seed whenever there is a random number generation to ensure reproducibility!
- Pandas slicing:
	- `df[]` should access only to column names/labels: `df['col_name']`.
	- `df.iloc[]` can access only to row & column index numbers + booleans: `df.iloc[0,'col_name']`, `df.iloc[:,'col_name']`.
	- `df.loc[]` can access only to row & column labels/names + booleans: `df.loc['a','col_name']`, `df.loc[:,'col_name']`.
	- `df[]` can be used for changing entire column values, but `df.loc[]` or `df.iloc[]` should be used for changing sliced row values.

## Data Cleaning

- Always do general checks: `df.head()`, `df.info()`, `df.describe()`, `df.shape`.
- Always get lists of column types: `select_dtypes()`.
	- Categorical columns, `object`: `unique()`, `value_counts()`. Can be subdivided in:
		- `str`: text or category levels.
		- `datetime`: encode with `to_datetime()`.
	- Numerical columns, `int`, `float`: `describe()`.
- Detect and remove duplicates: `duplicated()`, `drop_duplicates()`.
- Correct inconsistent text/typos in labels, use clear names: `replace()`, `map()`.
- Beware: many dataframe modifying operations require `inplace=True` flag to change the dataframe.
- Detect and fix missing data: 
	- `isnull() == isna()`.
	- Plot the missing amounts: `df.isnull().sum().sort_values(ascending=False)[:10].plot(kind='bar')`.
	- Analyze the effect of missing values on the target: take a feature and compute the target mean & std. for two groups: missing feature, non-missing feature.
		- This could be combined with a T-test.
	- Sometimes the missing field is information:
		- It means there is no object for the field; e.g., `license_povided`: if no string, we understand there is no license.
		- We can create a category level like `'Missing'`
		- We can mask the data: create a category for missing values, in case it leads to insights.
	- `dropna()` rows if several fields missing or missing field is key (e.g., target).
	- `drop()` columns if many (> 20-30%) values are missing. 
	- Impute the missing field/column with `fillna()` if few (< 10%) rows missing: `mean()`, `meadian()`, `mode()`.
	- More advanced:
		- Predict values with a model.
		- Use k-NN to impute values of similar data-points.
- Detect and handle outliers:
	- Linear models are shifted towards the outliers!
	- Keep in mind the empricial rule of 68-95-99.7 (1-2-3 std. dev.).
	- Compute `stats.zscore()` to check how many std. deviations from the mean; often Z > 3 is considered an outlier.
	- Histogram plots: `sns.histplot()`.
	- Box plots: `sns.boxplot()`.
	- Scatterplots: `plt.scatter()`, `plt.plot()`.
	- Residual plots: differences between the real/actual target values and the model predictions.
	- IQR calculation: use `np.percentile()`.
	- Drop outliers? Only if we think they are not representative.
	- Do transformations fix the `skew()`? `np.log()`, `np.sqrt()`, `stats.boxcox()`, `stats.yeojohnson()`.
		- Usually a absolute skewness larger than 0.75 requires a transformation (feature engineering).
- Temporal data / dates or datetime: they need to be converted with `to_datetime()` and the we need to compute the time (in days, months, years) to a reference date (e.g., today).


## Exploratory Data Analysis

- Recall we have 3 main ways of plotting:
	- Matplotlib: default: `plt.hist()`.
	- Seaborn: nicer, higher interface: `sns.histplot()`.
	- Pandas built-in: practical: `df.plot(kind='hist')`, `df.hist()`, `df.plot.hist()`; `plt` settings passed as arguments!
- Usually, the exploration is done plotting the independent variables (features) against the target (dependent or predicted variable).
- We need to plot / observe **every** feature or variable:
	- Create automatic lists of variable types: `select_dtypes()`. Usual types:
		- Numerical: `int`, `float`.
		- Strings: `object`. Can contain:
			- `str`: text or category level.
			- `datetime`: encode with `to_datetime()`.
	- Automatically created lists often need to be manually processed, especially `object` types. 
	- Loop each type list and apply the plots/tools we require.
- Quantitative/numerical variables: can be uni/multi-variate; most common EDA tools:
	- Histograms: `sns.histplot()`, `plt.hist()`, `df.hist()`.
		- Look at: shape, center, spread, outliers.
	- Numerical summaries: `describe()`.
	- Boxplots: `sns.boxplot()`.
		- Look at outliers.
		- Also, **combine these two**:
			- Boxplot: `sns.catplot()`.
			- Points overlapped: `sns.stripplot()`.
	- Scatteplots: `sns.scatterplot()`, `plt.scatter()`, `sns.regplot()`, `sns.lmplot()`
		- Look at: linear/quadratic relationship, positive/negative relationship, strength: weak/moderate/strong.
		- Beware of the [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox).
	- Correlations: `df.corr()`,  `stats.pearsonr()`; see below.
- Categorical variables: ordinal (groups have ranking) / cardinal (no order in groups); most common EDA tools:
	- Count values: `unique()`, `value_counts().sort_values(ascending=False).plot(kind='bar')`.
	- Frequency tables; see below.
	- Bar charts: `sns.barplot()`, `plt.bar()`, `plt.barh()`; use `sort_values()`.
	- Count plots: `sns.countplot()`.
- If a continuous variable has different behaviors in different ranges, consider stratifying it with `pd.cut()`. Example: `age -> age groups`.
- Frequency tables: stratify if necessary, group by categorical levels, count and compute frequencies.
	- Recipe: `groupby()`, `value_counts()`, normalize with `apply()`.
	- See also: `pd.crosstab()`.
- Correlations:
	- Heatmap for all numerical variables: `sns.heatmap(df.corr())`.
		- `cmap`: [Matplotlib colormaps](https://matplotlib.org/stable/gallery/color/colormap_reference.html).
		- [Seaborn color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html).
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
- Transformations:
	- General notes:
		- Apply if variables have a `skew()` larger than 0.75; we can also perform normality checks: `scipy.stats.mstats.normaltest`.
		- Try first how well work simple functions: `np.log()`, `np.log1p()`, `np.sqrt()`.
		- If `hist()` / `skew()` / `normalitytest()` don't look good, try power transformations, but remember saving their parameters and make sure we have an easi inverse function:
			- Box-Cox: generalized power transformation which usually requires `x > 0`: `boxcox = (x^lambda + 1)/lambda`.
			- Yeo-Johnson: more sophisticated, piece-wise - better results, but more difficult to invert & interpret.
	- Target: although it is not necessary for it to be normal, nomal targets yield better R2 values.
		- Often the logarithm is applied: `df[col] = df[col].apply(np.log1p)`.
		- That makes undoing the transformation very easy: `np.exp(y_pred)`.
		- However, check if power transformations are better suited (e.g., `boxcox`, `yeojohnson`); if we use them we need to save the params/transformer and make sure we know how to invert the transformation!
	- Predictor / independent variables:
		- `scipy` or `sklearn` can be used for power transformations, e.g., `boxcox`, `yeojohnson`.
		- We can discretize very skewed variables, i.e., we convert then into categorical: we transform the distributions into histograms in which bins are defined as equal width/frequency. That way, each value is assigned the bin number. Additionally, see binarization below.
- Extract / create new features, more descriptive:
	- Multiply different features if we suspect there might be an interaction.
	- Divide different features, if the division has a meaning.
	- Create categorical data from continuous if that has a meaning, e.g., daytime.
	- Try polynomial features: `PolynomialFeatures()`.
	- Create deviation factors from the mean of a numeric variable in groups or categories of another categorical variable. 
- Measure the cardinality of the categorical variables: how many catgeories they have.
	- `data[cat_vars].nunique().sort_values(ascending=False).plot.bar(figsize=(12,5))`.
	- Tree-based models overfit if we have
		- many categories,
		- rare labels.
	- Replace categorical levels with few counts (rare) with `'other'`.
- Categorical feature encoding:
	- One-hot encoding / dummy variables: `get_dummies()`.
		- Alternative: `sklearn.preprocessing.OneHotEncoder`.
		- In general, `sklearn` encoders are objects that can be saved and have attributes and methods: `classes_`, `transform()`, `inverse_transform()`, etc.
	- Binarization: manually with `apply()`, `np.where()` or `sklearn.preprocessing.LabelBinarizer`.
		- Usually very skewed variables are binarized.
			- We can check the predictive strength of binarized variables with bar plots and T tests: we binarize and compute the mean & std. of the target according to the binary groups.
		- `LabelBinarizer`: `fit([1, 2, 6, 4, 2]) -> transform([1, 6]): [[1, 0, 0, 0], [0, 0, 0, 1]]`.
		- Sometimes it is necessary to apply this kind of multi-class binarization to the **target**.
	- Also useful for the **target**: `sklearn.preprocessing.LabelEncoder`: `fit(["paris", "paris", "tokyo", "amsterdam"]) -> transform(["tokyo", "paris"]): [2, 1]`.
	- Ordinal encoding: convert ordinal categories to `0,1,2,...`; but be careful: we're assuming the distance from one level to the next is the same -- Is it really so? Maybe it's better applying one-hot encoding?
		- `sklearn` tools: `OrdinalEncoder`, `DictVectorizer`.
	- There are many more tools: [Preprocessing categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features)
	- Weight of evidence: If the target is binary and we want to encode categorical features, we can store the target ratios associated to each feature category level.
- Train/test split: perform it before scaling the variables: `train_test_split()`.
- Feature scaling: apply it if data-point distances are used in the model; fit the scaler only with the train split!
	- `StandardScaler()`: subtract the mean and divide by the standard deviation; features are converted to standard normal viariables. Note that if dummy variables `[0,1]` passed, they are scaled, too. That should not be an issue,  but the interpretation is not as intuitive later on. Alternatives: use `MinMaxScaler()` or do not pass dummies to the scaler.
	- `MinMaxScaler()`: a mapping with which the minimum value becomes 0, max becomes 1. This is senstive to outliers!
	- `RobustScaler()`: IQR range is mapped to `[0,1]`, i.e., percentiles `25%, 75%`; thus, the scaled values go out from the `[0,1]` range.

### Scikit-Learn Transformers

List of the most important Scikit-Learn Transformers:

- [Missing data imputation](https://scikit-learn.org/stable/modules/impute.html#impute)
  - `sklearn.impute.SimpleImputer`: we define what is a missing value and specify a strategy for imputing it, e.g., repleace with the mean.
  - `sklearn.impute.IterativeImputer`: features with missing values are modelled with the other features, e.g., a regression model is built to predict the missing values.
- [Categorical Variable Encoding](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features)
  - `sklearn.preprocessing.OneHotEncoder`: dummy variables of all levels (except one) in a categorical variable are created, i.e., a binary variable for each category-level.
  - `sklearn.preprocessing.OrdinalEncoder`: string variables are converted into ordered integers; however, be careful, because these cannot be used in scikit-learn if they do not really represent continuous variables... if that is not the case, try the `OneHotEncoder` instead.
- [Scalers](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)
  - `sklearn.preprocessing.MinMaxScaler`: data mapped to the min-max range
  - `sklearn.preprocessing.StandardScaler`: substract mean and divide by standard deviation
  - `sklearn.preprocessing.RobustScaler`: scaling with the IQR done
  - ...
- [Discretisation](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization)
  - `sklearn.preprocessing.KBinsDiscretizer`: quantization, partition of continuous variables into discrete values; different strategies available: constant-width bins (uniform), according to quantiles, etc.
- [Variable Transformation](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-transformer)
  - `sklearn.preprocessing.PowerTransformer`: Yeo-Johnson, Box-Cox
  - `sklearn.preprocessing.FunctionTransformer`: It constructs a transformer from an arbitrary callable function
  - ...
- [Variable Combination](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features)
  - `sklearn.preprocessing.PolynomialFeatures`: given a degree d, obtain polynomial features up to the degree: x_1, x_2, d=2 -> x_1, x_2, x_1*x_2, x_1^2, x_2^2
- [Text Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
  - `sklearn.feature_extraction.text.CountVectorizer`: create a vocabulary of the corpus and pupulate the document-term matix `document x word` with count values.
  - `sklearn.feature_extraction.text.TfidfTransformer`: create the document-term matrix by scaling with in-document an in-corpus frequencies.

### Creation of Transformer Classes

Manual definition:

```python
# Parent class: its methods & attributes are inherited
class TransformerMixin:
    def fit_transform(self, X, y=None):
        X = self.fit(X, y).transform(X)
        return X

# Child class
# Same class definition as before
# BUT now we inherit the methods and attributes
# from TransformerMixin
class MeanImputer(TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y=None):
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self
    def transform(self, X):
        for v in self.variables:
            X[v] = X[v].fillna(self.imputer_dict[v])
        return X

# Usage
my_imputer = MeanImputer(variables=['age','fare'])
my_imputer.fit(X) # means computed and saved as a dictionary
X_transformed = my_imputer(X) # We get the transformed X: mean imputed in NA cells
X_transformed = my_imputer.fit_transform(X) 
my_imputer.variables # ['age','fare']
my_imputer.imputer_dict_ # {'age': 39, 'fare': 100}
```

Classes derived from `sklearn`; they have the advantage that they can be stacked in a `Pipeline`:

```python
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class MeanImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # Learn and persist mean values in a dictionary
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
```

## Feature Selection

- Less features prevent overfitting and are easier to explain.
- Remove features which almost always (99% of the time) have the same value.
- There are three major approaches for feature selection:
	1. The effect of each variable is analyzed on the target using ANOVA or similar.
	2. Greedy approaches: all possible feature combinations are tested (very expensive).
	3. Lasso regularization: L1 regularization forces coefficients of less important variables to become 0; thus, we can remove them. This is the method I have normally used.
- We can measure sparsity of information with `PCA()`; if less variables explain most of the variance, we could drop some.
- Typical method: Select variables with L1 regularized regression (lasso): `SelectFromModel(Lasso())`.
- Use `sns.pairplot()` or similar to check multi-colinearity; correlated features are not good.
- If the model is overfitting, consider dropping features.
	- Ovefitting: when performance metric is considerably better with train split than in cross-validation/test split.

## Hypothesis Tests

- Always define `H0`, `Ha` and `alpha` beforehand and keep in mind the errors:
	- Type I error: you're healthy but the test says you're sick: False positive.
	- Type II error: your're sick, but the test says you're healthy: False negative.
	- P(Type I error) = `alpha`, significance level, typically 0.05.
	- P(Type II error) = `beta`.
	- Power of a test = 1 - `beta`: depends on `alpha`, sample size, effect size.
		- We can estimate it with a **power analysis**.
- Most common hypothesis tests
	- Two independent proportions: Z Test (Standard Distribution).
	- Two independent means: T Test (Stundent's T Distribution).
	- One factor with L>2 levels creating L means: One-way ANOVA (Fisher's F Distribution).
	- One factor with L>2 levels creating L proportions/frequencies: Chi Square Test with contingency table.
		- Example contingency table: `[age_group, smoke] = (18-30, 31-50, 51-70, 71-100) x (yes, no)`.
		- Use `crosstab.
- Take into account all the assumptions made by each test and the details!
	- Independent vs. paired groups (repeated measures; within-studies).
	- One-sided (H: >, <) vs 2-sided tests (H: !=).
	- Normality assumption: check histograms, QQ plots, run normality tests if necessary.
	- Equal variances assumption.
	- If assumptions broken, consider equivalent parametric tests.
	- Post-hoc tests when >2 levels (e.g., after ANOVA): apply Bonferroni correction if T tests used: `alpha <- alpha / num_tests`.

## Data Modelling

Data modelling is out of the scope of this guide, because the goal is to focus on the data processing and analysis part prior to creating models. However, some basic modelling steps are compiled, since they often provide feedback for new iterations in the data processing.

- Most common approaches to start with tabular data:
	- Supervised learning:
		- Regression: `Ridge`, `RandomForestRegressor`.
		- Classification: `RandomForestClassifier`.
	- Unsupervised learning:
		- Clustering: `KMeans`.
		- Dimensionality reduction: `PCA`.
- Always evaluate with cross-validation/test split.
	- Regression: R2, RMSE.
	- Classification: confusion matrix, accuracy, F1, ROC curve (AUC).
- Plot model parameters to understand what's going on; often it's better than the predictions.
- If text reports need to be saved, convert them to figures: `plt.text()`.

## Tips for Production

Software engineering for production deployments is out of the scope of this guide, because the goal is to focus on the data processing and analysis; however, some minor guidelines are provided. Going from a research/development environment to a production environment implies we need to:

- assure reproducibility,
- apply all checks, encodings and transformations to new data points on-the-fly,
- score the model as new data arrives seamlessly and track the results.

Thus, among others, we should do the following:

- Persist any transformation objects / encodings / paramaters generated.
- Track and persist any configuration we created.
- Use seeds whenever any random variables is created.
- Create python environments and use same software versions in research/production.
	- Containers are a nice option.
- Use `Pipelines` and pack any transformations to them. That implies:
	- Using `sklearn` transformers/encoders instead of manually encoding anything,
	- or `feature_engine` classes: [feature_engine](https://feature-engine.readthedocs.io/en/latest/),
	- or creating our own functions embedded in derived classes of these, so that they can be added to a `Pipeline`.
- Modularize the code into functions to transfer it to python scripts.
- Catch errors and provide a robust execution.
- Log events.
- Go for Test Driven Development (TDD).
- Use code and dataset and version control.
- Monitor the results of the model.

### Pipelines

In the following, a vanilla example with `Pipeline`:

```python
# Import necessary Transformers & Co.
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)

# Add sequential steps to the pipeline: (step_name, class)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

# The pipeline can be used as any other estimator
pipe.fit(X_train, y_train)

# Inference with the Pipeline
pred = pipe.predict(X_test)
pipe.score(X_test, y_test)
```

## Relevant Links

- My notes on the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) from Coursera: [machine_learning_ibm](https://github.com/mxagar/machine_learning_ibm).
- My notes on the [Statistics with Python Specialization](https://www.coursera.org/specializations/statistics-with-python) from Coursera (University of Michigan): [statistics_with_python_coursera](https://github.com/mxagar/statistics_with_python_coursera).
- My forked repository of the Udemy course [Deployment of Machine Learning Models](https://www.udemy.com/course/deployment-of-machine-learning-models/) by Soledad Galli and Christopher Samiullah: [deploying-machine-learning-models](https://github.com/mxagar/deploying-machine-learning-models).
- An example where I apply some of the techniques explained here: [airbnb_data_analysis](https://github.com/mxagar/airbnb_data_analysis).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You can freely use and forward this repository if you find it useful. In that case, I'd be happpy if you link it to me :blush:.
