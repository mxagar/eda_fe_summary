'''This python script summarizes many code snippets for data processing.
The file won't run! It is a compilation of important code pieces ready
for copy & pasting (& adapting).

See the companion README.md with schematic explanations.

Table of contents:

- Imports
- Loading and General, Useful & Important Functions
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Inferences & Hypothesis Testings
- Data Modeling (Supervised Learning)
- Dataset Structure (Unsupervised Learning)

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

from sklearn.metrics import classification_report, confusion_matrix, roc_curve #, plot_roc_curve
from sklearn.metrics import precision_recall_fscore_support as classification_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, plot_confusion_matrix

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

# Plotting styles can be modified!
sns.set_context('talk')
sns.set_style('white')

##### -- 
##### -- Data Ingestion/Loading and General, Useful & Important Functions
##### -- 

# Reading CSV
df = pd.read_csv('data/dataset.csv')
# If messy data, and columns might have many types, convert them to str: dtype=str
# If first rows are info, not CSV: skiprows
df = pd.read_csv('data/dataset.csv', dtype=str, skiprows=4)
# Other options:
# parse_dates=['col_name']
# index_col='col_name'
# names=column_names
# na_values='?'

# If tehre is a vectors/embedding column, we need to convert it
import ast
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Writing CSV
df.to_csv('data/dataset.csv', sep=',', header=True, index=False)
# Printing
df.to_string()
df.to_html()

# Read from JSON
# Note: this approach reads a JSON file
# Important to select orient depending on JSON formatting:
# https://pandas.pydata.org/docs/reference/api/pandas.read_json.html
df = pd.read_json('population_data.json', orient='records')
df.to_json('population_data.json', orient='records')

# API + JSON object
import requests
# Define URL: Rural population in Switzerland
url = 'http://api.worldbank.org/v2/country/ch/indicator/SP.RUR.TOTL/?date=1995:2001&format=json&per_page=1000'
# Send the request
r = requests.get(url)
# Convert to JSON: first element is metadata
r_json = r.json()
df = pd.DataFrame(r_json[1])
# Now cleaning is necessary...

# Read from XML
# Often more manual processing is required
# Example XML file with these "record" objects and "fields" within:
# <record>
#   <field name="Country or Area" key="ABW">Aruba</field>
#   <field name="Item" key="SP.POP.TOTL">Population, total</field>
#   <field name="Year">1960</field>
#   <field name="Value">54211</field>
# </record>
# Parse with BeautifulSoup
from bs4 import BeautifulSoup
with open("population_data.xml") as fp:
    soup = BeautifulSoup(fp, "lxml") # lxml is the Parser type
# Convert the XML into dataframe
data_dictionary = {'Country or Area':[], 'Year':[], 'Item':[], 'Value':[]}
for record in soup.find_all('record'): # look for "record" objects
    for record in record.find_all('field'): # look for "field" objects
        data_dictionary[record['name']].append(record.text)
df = pd.DataFrame.from_dict(data_dictionary)
#   Country or Area	 Year	 Item	               Value
# 0	Aruba	           1960	 Population, total	 54211
# ...
# We need to / can pivot the table for better format
df = df.pivot(index='Country or Area', columns='Year', values='Value')
df.reset_index(level=0, inplace=True)
#  	Country or Area	  1960	    1961	    1962	    1963	1964	...	2017
# 0	Afghanistan	      8996351	  9166764	  9345868	  ...
# ...

# Parse HTML: Example parse course title + description from web
# Note that if the content is generated by Javascript, it won't work:
# https://stackoverflow.com/questions/8049520/web-scraping-javascript-page-with-python
import requests 
from bs4 import BeautifulSoup
r = requests.get('https://learndataengineering.com/p/all-courses')
print(r.text) # raw text, useless
# Parse content
#soup = BeautifulSoup(r.text, "lxml")
soup = BeautifulSoup(r.content, 'html.parser')
print(soup.get_text()) # parsed text, useless
# Get all courses: title, description
# We need to right click + inspect to see the name of the CSS object
# of each course card; then, we need to see the hierarchical components
# of that object which contain the title and the description
course_objects = soup.find_all("div", {"class": "featured-product-card__content"})
courses = []
for course in course_objects:
    title = course.select_one("h3").get_text().strip()
    description = course.select_one("h4").get_text().strip()
    courses.append((title, description))

# Read from SQL
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
# with SQLite
# Connect to the database
conn = sqlite3.connect('population_data.db')
# Run a query: there is only one table, population_data, and we extract everything
df = pd.read_sql('SELECT * FROM population_data', conn)
# with SQLAlchemy
# Create an engine
engine = create_engine('sqlite:///population_data.db')
# Run SELECT * query
# WATCH OUT: sqalchemy<2.0
df = pd.read_sql("SELECT * FROM population_data", engine)

# Write to SQLite using sqlite3
# Alternative: SQLAlchemy
# Look also: https://github.com/mxagar/sql_guide
import sqlite3
# Connect to database; file created if not present
conn = sqlite3.connect('dataset.db')
# Load Table A - or create one
df_A = pd.read_csv('dataset_A.csv')
# Load Table B - or create one
df_B = pd.read_csv('dataset_B.csv')
# Clean, if necessary
columns = [col.replace(' ', '_') for col in df_A.columns]
df_A.columns = columns
# ...
# Write tables to database
df_A.to_sql("table_A", conn, if_exists="replace", index=False)
df_B.to_sql("table_B", conn, if_exists="replace", index=False)
# Check (i.e., read)
df_A_ = pd.read_sql('SELECT * FROM table_A', conn)
df_B_ = pd.read_sql('SELECT * FROM table_B', conn)
# Commit changes and close connection
conn.commit()
conn.close()

# Insert rows to SQLite
# WARNING: use better to_sql() and pass entire tables
# i.e., don't insert row-by-row in a for loop...
# Connect to the data base, create if file not there
conn = sqlite3.connect('database.db')
# Get a cursor
cur = conn.cursor()
# Drop the test table in case it already exists
cur.execute("DROP TABLE IF EXISTS test")
# Create the test table including project_id as a primary key
cur.execute("CREATE TABLE test (project_id TEXT PRIMARY KEY, countryname TEXT, countrycode TEXT, totalamt REAL, year INTEGER);")
# Insert a single row of value into the test table
project_id = 'a'
countryname = 'Brazil'
countrycode = 'BRA'
totalamt = '100,000'
year = 1970
sql_statement = f"INSERT INTO test (project_id, countryname, countrycode, totalamt, year) VALUES ('{project_id}', '{countryname}', '{countrycode}', '{totalamt}', {year});"
cur.execute(sql_statement)
# Commit changes made to the database
conn.commit()
# Select all from the test table
cur.execute("SELECT * FROM test")
cur.fetchall()
# [('a', 'Brazil', 'BRA', '100,000', 1970)]
# Insert several rows:
for index, values in df.iterrows():
    project_id, countryname, countrycode, totalamt, year = values
    sql_statement = f"INSERT INTO test (project_id, countryname, countrycode, totalamt, year) VALUES ('{project_id}', '{countryname}', '{countrycode}', '{totalamt}', {year});"
    cur.execute(sql_string)
# Commit changes to the dataset after any changes are made
conn.commit()
# ...
# Commit changes and close connection
conn.commit()
conn.close()

## Progressbar: visualize progress in heavy loops
# https://progressbar-2.readthedocs.io/en/latest/
# pip install progessbar
import progressbar
counter = 0
bar = progressbar.ProgressBar(maxval=len(items)+1, # number of expected counter steps
                              term_width=50, # width of output
                              widgets=[progressbar.Bar('=', '[', ']'), # see output below
                                       '->',
                                       progressbar.Percentage()])
bar.start()
    
for item in items:
    # Update the progress bar
    counter+=1 
    bar.update(counter)
    # DO HEAVY WORK with item
    # ...
    
bar.finish()
# [==========================================]->100%

# Fetch/extract from HTML tables
# That works when we have page with a clear HTML table in it: <table>...
year = 2019
url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
html = pd.read_html(url, header = 0)
df = html[0] # Take first table
# Then, we'll require some processing and cleaning: drop(), fillna(0), ...

# Get from web as ZIP; extract and load it
import requests
import zipfile
import io
content = requests.get(
    "https://archive.ics.uci.edu/.../Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("day.csv"), 
                           header=0, 
                           sep=',')

# Import a python list
import ast
with open('previous_scores.txt', 'r') as f:
    scores_list = ast.literal_eval(f.read())

# Read from TXT file
with open("population_data.txt", 'r') as file:
    # Example: pick first 10 lines
    lines = file.readlines()[:10]
    for index, line in enumerate(lines):
        print(index, line)

# Serialize and save any python object, e.g., a model/pipeline
# BUT, WARNING: 
# - Python versions must be consistent when saving and loading.
# - If we use lambdas as part of pipeline elements, these might have
# problems while being serialized as pickles; instead, use function definitions
# which can be imported and/or skops as serializer!
import pickle
pickle.dump(model, open('model.pickle','wb')) # wb: write bytes
model = pickle.load(open('model.pickle','rb')) # rb: read bytes
# skops: https://skops.readthedocs.io/en/stable/
# python -m pip install skops
import skops.io as sio
with open(pipe_filename, 'wb') as f:
    sio.dump(model_pipe, f)
with open(pipe_filename, 'rb') as f:
    model_pipe = sio.load(f, trusted=True)

# Save ASCII files, e.g. TXT
lines = ['Readme', 'How to write text files in Python']
with open('readme.txt', 'w') as f:
    f.writelines(lines) # lines separated by \n

with open('readme.txt', 'w') as f:
    f.write('\n'.join(lines)) # equivalent to previous
    
more_lines = ['', 'Append text files', 'The End']
with open('readme.txt', 'a') as f:
    f.write('\n'.join(more_lines)) # append to an existing file

# Get current date & time
from datetime import date, datetime
print(date.today()) # 2023-01-17
now = datetime.now()
print(now) # 2023-01-17 09:12:57.731891
print(now.strftime("%d/%m/%Y %H:%M:%S")) # 17/01/2023 09:12:57

# General info on dataframe structure and columns
df.head(3)
df.info()
df["price"].describe() # use .T if many columns
df.dtypes # columns + type
df.dtypes.value_counts() # counts of each type

# Convert matrix/dict <-> dataframe
df_matrix = np.matrix(df.head(1000)) # first 1000 entries
df_matrix = np.matrix(df)
df_matrix = df.values
df = pd.DataFrame(data=np.ones((3,3)), columns=['a', 'b', 'c']) # data: np.ndarray
df = pd.DataFrame(data={'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]}) # data: dict
df.to_dict(orient="list") # experiment with orient

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

# Unique values with numpy + count
unique_dates, count = np.unique(dates, return_counts=True)

# Set operations with unique elements
list(set(df_outliers_population['country']).intersection(df_outliers_gdp['country']))
list(set(df_outliers_population['country']) - set(df_outliers_gdp['country']))
# Intersection operation with numpy using arrays
np.intersect1d(movies1, movies2, assume_unique=True)

# Cast a variable / dataframe
df['var'] = df['var'].astype('O')
df = df.astype('int32')
# Cast a string number with , thousand separators to numeric
df['value'] = df['totalamt'].str.replace(',', '')
df['value'] = pd.to_numeric(df['value'])

# Tukey rule: 1.5*(q75-q25) = 1.5*IQR -> outlier detection
q25, q50, q75 = np.percentile(df['price'], [25, 50, 75])
# Skewness
df['price'].skew()
# Skewness: an absolute value larger than 0.75 requires transformations
for col in numerical_cols:
    if np.abs(df[col].skew()) > 0.75:
        df[col] = np.log1p(df[col])

# Get uniques and counts of the levels of a categorcial variable
df["condition"].value_counts().sort_values(ascending=False).plot(kind='bar')
df["condition"].unique()

# Aggregate functions
df['col'].max()
df['col'].min()
df['col'].mean()
df.max() # axis = 0, 1
df['col'].idxmax()
df['col'].idxmin()

# Pandas sample of n=5 rows/data-points that only appear once
sample = df.sample(n=5, replace=False)

# Basic text processing
word_list = text.lower().split(' ')
number = int(text.split(' ')[0])

# Apply string operations to string column values
# https://github.com/mxagar/disaster_response_pipeline/blob/main/disaster_response/process_data.py
# Example 1: split text content in columns a;b;c -> a | b | c
df.col.str.split(pat=";", n=-1, expand=True)
# Example 2: split content in - and take second value: 'related-1' -> '1'
df[col] = df[col].str.split('-').str.get(1)

# Group By
# When grouping by a column/field,
# we apply the an aggregate function
df.groupby('species').mean()
df.groupby('species').agg([np.mean, np.median])
df.groupby(['year', 'city'])['var'].median()
df.groupby('species').size() # count
# Often we want to convert a groupby series to a dataframe
# and then perform a join
trees = df.groupby('district_name').size()
trees = trees.to_frame(name='num_trees') # convert series to dataframe
pd.merge(districts)

# Average job satisfaction depending on company size
df.groupby('CompanySize')['JobSatisfaction'].mean().dropna().sort_values()

# Dates: are usually 'object' type, they need to be converted & processed
# Format:
#   https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
#   https://strftime.org
df['date'] = pd.to_datetime(df['date'], format='%b-%y') # format='%Y-%m-%d'
df['month'] = df['date'].dt.month # better: month_name()
df['month'] = df['date'].dt.month_name().str.slice(stop=3)
df['weekday'] = df['date'].dt.weekday
df['year'] = df['date'].dt.year
# Time delta: Duration, to_timedelta
# https://pandas.pydata.org/docs/reference/api/pandas.to_timedelta.html
df['duration'] = pd.to_timedelta(df['duration'])
# Time comparisons / filterings: date might need to be converted with pd
import datetime as dt
d = pd.to_datetime(dt.date(1992, 4, 27), utc=True) # sometimes UTC is an issue, eg., when comparing
df_before =  df[(df['date'] < d)]

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
# If iloc returns a Series, to make it return a single-row dataframe, use [[]]
df.iloc[3, :] # Series of row with index value 3
df.iloc[[3]] # Single-row dataframe or row with index 3
df.iloc[[2,3]] # Dataframe with rows 2 & 3

# Index: it might contain information like column!
df.index.to_list()

# Iterate the rows of a dataframe
for index, row in df.iterrow():
    # Extract columns
    var1, var2, var3, var4 = row

# Selection / Filtering: Column selection after name
col_names = df.columns # 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXLEG', 'BMXARML', ...
keep = [column for column in col_names if 'BMX' in column]
df_BMX = df[keep]

# Selection / Filtering: With booleans
df.loc[:, keep].head()
# Indexing with booleans: Which column (names) are in keep?
index_bool = np.isin(df.columns, keep)
df_BMX = df.iloc[:,index_bool]

# Selection / Filtering: Pandas isin() +  multiple conditions
# isin() can be handy for categorical variables
selected_teams = ['A', 'B']
selected_positions = ['left', 'right']
df_filtered = df[(df.team.isin(selected_teams) & df.position.isin(selected_positions))]

# Filtering with between
idx = df['price'].between(min_price, max_price)
df = df[idx].copy()

# Filter and convert to list
var2_large_list = df[df.var1 > 1.0]['var2'].to_list()

# Filtering with text: .str.contains()
# Country names can change, sometimes we need to filter non-official names, e.g.:
# Yogoslavia, Former Yogoslavia, etc.
df[df.countryname.str.contains('Yugoslavia')]

# Multiple filtering: Several conditions
waist_median = df_BMX['BMXWAIST'].median()
condition1 = df_BMX['BMXWAIST'] > waist_median
condition2 = df_BMX['BMXLEG'] < 32
df_BMX[condition1 & condition2].head()

# Multiple filtering: Another example
df_filtered = df[(df['location'] == "Munich, Germany") | (df['location'] == "London, England")]
cities = ['Munich', 'London', 'Madrid']
df_filtered = df[df.location.isin(cities)]

# Filtering: queries
value1 = 1
value2 = 2
df.query('col1 == @value1 and col2 > (@value2 -1)')

## Combining datasets: concat
df = pd.concat([X,y],axis=1) # columns after columns, same rows
df = pd.concat([df1,df2]) # rows after rows, same columns

## Combining datasets: Joins/Merges
# More on JOINS: https://www.w3schools.com/sql/sql_join.asp
#
# Example:
# popular_courses_df: pd.DataFrame
#       course	enrollments
#   0   CS01    1200
#   1   MA03    1000
#   2   CS02    850
# ...
# courses_df: pd.DataFrame
#       course_id   title                           classroom
#   0   PH01        Physics I                       B01R01
#   1   MA01        Algebra                         B01R01
#   2   CS01        Introduction to Programming     B02R05
# ...
#
# We want to have the title/name of the course in the
# dataframe popular_courses.
# We need to do an inner join! i.e., intersecting values taken
popular_courses_ = pd.merge(left=popular_courses_df,
         right=courses_df,
         how='inner',
         left_on='course',
         right_on='course_id')[['course', 'enrollments', 'title']]
# popular_courses_
#       course	enrollments title
#   0   CS01    1200        Introduction to Programming
#   1   MA03    1000        Calculus II
#   2   CS02    850         Data Structures and Algorithms

## Combining datasets
# Merging datasets: OUTER JOINS
# More on JOINS:
# https://www.w3schools.com/sql/sql_join.asp
# https://guides.nyu.edu/quant/merge
#
# Example: 
# df1 and df2 have same columns: col1, col2
# Some rows appear only in df1, some only in df2, some in both
# We want to merge both: we need an OUTER JOIN
# AND we can informatively mark where each row came from
# with indicator=True
df_all = df1.merge(df2.drop_duplicates(),
                   on=['col1','col2'],
                   how='outer', 
                   indicator=True)
# df_all
#       col1    col2    _merge
#   0   7       90      both
#   1   6       81      left_only
#   2   2       72      right_only
#   3   9       63      both
#   ...

## Flattening of arrays/lists
# to create dataframes
import itertools
id_num = [['doc1','doc1','doc1'], ['doc2','doc2','doc2']]
token = [['hello','my','name'], ['hello','your','bye']]
count = [[2,3,1], [1,1,2]]
id_num_ = list(itertools.chain(*id_num)) # ['doc1','doc1','doc1','doc2','doc2','doc2']
token_ = list(itertools.chain(*token)) # ['hello','my','name','hello','your','bye']
count_ = list(itertools.chain(*count)) # [2,3,1,1,1,2]
data_dict = {"id": id_num_,
             "token": token_,
             "count": count_}
df = pd.DataFrame(data_dict)
#
# 	    id	    token	count
# 0	    doc1	hello   2
# 1 	doc1	my	    3
# 2	    doc1	name	1
# 3	    doc2	hello	1
# 4	    doc2	your	1
# 5 	doc2	bye	    2

## Pivoting (see also the next example)
# Re-arrange previous dataframe df
df_slice = df[df['id'] == 'doc1']
df_slice_T = df_slice.pivot(index=['id'], columns='token').reset_index(level=[0])
#
#
#	    id	                    count
# token		    hello	my	    name
# 0	    doc1	2	    3	    1
df_slice_T.iloc[0,1:].values # [2, 3, 1]

## Pivoting: Another example
# In this example a dense rating matrix is
# converted to a sparse user-item matrix
# Origin: 
# https://github.com/mxagar/machine_learning_ibm/blob/main/06_Capstone_Project/lab/lab_jupyter_cf_knn.ipynb
#
# rating_dense_df:
# 	user	item	    rating
# 0	1889878	CC0101EN	3.0
# 1	1342067	CL0101EN	3.0
# 2	1990814	ML0120ENv3	3.0
# 3	380098	BD0211EN	3.0
# 4	779563	DS0101EN	3.0
rating_sparse_df = rating_dense_df.pivot(index='user', columns='item', values='rating').fillna(0).reset_index().rename_axis(index=None, columns=None)
rating_sparse_df.head()
# 	user	AI0111EN	BC0101EN	BC0201EN ...
# 0	2	    0.0	        3.0	        0.0	     ...
# 1	4	    0.0	        0.0	        0.0	     ...
# 2	5	    2.0	        2.0	        2.0	     ...
# 3	7   	0.0	        0.0	        0.0	     ...
# 4	8	    0.0	        0.0	        0.0	     ...

## Unpivoting a dataset from wide to long format: melt
# Example:
# https://github.com/mxagar/data_science_udacity/blob/main/03_DataEngineering/lab/05_combine_data/5_combining_data.ipynb
#
# df_rural.columns = 'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', ..., '2017'
df_rural = pd.read_csv('rural_population_percent.csv', skiprows=4)
# df_electricity.columns = 'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', ..., '2017'
df_electricity = pd.read_csv('electricity_access_percent.csv', skiprows=4)
# New format: long
# df_rural.columns = 'Country Name, 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'Rural Value'
df_rural = pd.melt(df_rural, id_vars=['Country Name',
                                      'Country Code',
                                      'Indicator Name',
                                      'Indicator Code'],
                             var_name='Year',
                             value_name='Rural Value')
# df_electricity.columns = 'Country Name, 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'Electricity Value'
df_electricity = pd.melt(df_electricity, id_vars=['Country Name',
                                                  'Country Code',
                                                  'Indicator Name',
                                                  'Indicator Code'],
                                         var_name='Year',
                                         value_name='Electricity Value')
# Drop any columns from the data frames that aren't needed
df_rural.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)
df_electricity.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)
# Merge the data frames together based on their common columns
# in this case, the common columns are Country Name, Country Code, and Year
df_merge = df_rural.merge(df_electricity, how='outer',
                                          on=['Country Name', 'Country Code', 'Year'])
# Sort the results by country and then by year
# df_combined.columns = 'Country Name', 'Country Code', 'Indicator Name',' Indicator Code', 'Year', 'Rural Value'
df_combined = df_merge.sort_values(by=['Country Name', 'Year'])

## Adding rows to a DataFrame
# Define a new row
new_row = {'col1': 'Monday', 
           'col2': 1.5, 
           'col3': 'A'}
# Append new row
df = df.append(new_row, ignore_index=True)

## Parsing arguments in "__main__"
import argparse
# Create parser
parser = argparse.ArgumentParser(description="ETL and Training Pipelines")
parser.add_argument("--config_filepath", type=str, required = False,
                    help="File path of the configuration file.")
# ... (more args)
# Parse arguments
args = parser.parse_args()
# Check arg and catch its value
config_file = "./config.yaml"
if args.config_filepath:
    config_file = args.config_filepath

##### -- 
##### -- Data Cleaning
##### -- 

## Text encodings
from encodings.aliases import aliases
# When an encoding is not UFT-8, how to detect which encoding we should use?
# Python has a file containing a dictionary of encoding names and associated aliases
alias_values = set(aliases.values())
for alias in alias_values:
    try:
        df = pd.read_csv('mystery.csv', encoding=alias)
        print(alias) # valid encodings are printed
    except:
         pass 
# Another option: chardet
# !pip install chardet
import chardet
with open("mystery.csv", 'rb') as file:
    print(chardet.detect(file.read())) # Encoding, confidence and language printed

# Rename column names
# - remove preceding blank space: ' education' -> 'education', etc.
# - replace - with _: 'education-num' -> 'education_num', etc.
df = df.rename(
    columns={col_name: col_name.replace(' ', '') for col_name in df.columns})
df = df.rename(
    columns={col_name: col_name.replace('-', '_') for col_name in df.columns})

# Remove or strip blank spaces from categorical column fields
categorical_cols = list(df.select_dtypes(include = ['object']))
for col in categorical_cols:
    df[col] = df[col].str.replace(' ', '')
# Alternatives:
# df[col] = df[col].str.strip()
# df = pd.read_csv('dataset.csv', skipinitialspace = True)   

# Get duplicated rows
# df.duplicated(['id']) -> False, False, ...
duplicate = df[df.duplicated(['id'])]
# Drop duplicates
duplicated_removed = df.drop_duplicates().reset_index(drop=True)
# Check that all indices are unique
df.index.is_unique

# Columns/Feature with NO missing values
no_nulls = set(df.columns[df.isnull().sum()==0])
# Columns/Feature with more than 75% of values missing
most_missing_cols = set(df.columns[(df.isnull().sum()/df.shape[0]) > 0.75])
# Missing values in each row
df.isnull().sum(axis=1)
# Count number of non-NaN values in a np.ndarray
num_values = np.count_nonzero(~np.isnan(matrix))

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
# WARNING: instead of imputing the column aggregate,
# we should group by other categorical features and impute the aggregate of that group!
df["var_fill"] = df.groupby("var_group")["var_fill"].transform(lambda x: x.fillna(x.mean()))
# Also, note more options:
# pandas.to_numeric: errors=‘coerce’: invalid parsing will be set as NaN
# pandas.mean(skipna=True): default is True
df['variable'].fillna(pd.to_numeric(df['variable'], errors='coerce').mean(skipna=True), inplace=True)

# Imputation: More options
fill_mode = lambda col: col.fillna(col.mode()[0]) # mode() returns a series, pick first value
df = df.apply(fill_mode, axis=0)
# BUT: Prefer better this approach
# because apply might lead to errors
num_vars = df.select_dtypes(include=['float', 'int']).columns
for col in num_vars:
    df[col].fillna((df[col].mean()), inplace=True)

# Imputation in time series: Forward Fill and Backward Fill
# i.e., if the data is ordered in time, we apply *hold last sample* 
# in one direction or the other. BUT: we need to sort the data!
df['GDP_ffill'] = df.sort_values(by='year').groupby("country")['GDP'].fillna(method='ffill')
df['GDP_bfill'] = df.sort_values(by='year').groupby("country")['GDP'].fillna(method='bfill')
# If only a country
df['GDP_ffill'] = df.sort_values(by='year')['GDP'].fillna(method='ffill')
# If the first/last value is NA, we need to run both: ffill and bfill
df['GDP_ff_bf'] = df.sort_values(by='year')['GDP'].fillna(method='ffill').fillna(method='bfill')

# Cleaning categories with Regex
# Fields with value '!$10' -> NaN
df['sector'] = df['sector'].replace('!$10', np.nan)
# Replace with Regex
# This looks for string with an exclamation point followed by one or more characters
df['sector'] = df['sector'].replace('!.+', '', regex=True)
# Replace with Regex
# Remove the string '(Historic)' from the sector1 variable
df['sector'] = df['sector'].replace('^(\(Historic\))', '', regex=True)
# More on regex:
# - Tutorial: https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285
# - Cookbook: https://medium.com/@fox.jonny/regex-cookbook-most-wanted-regex-aa721558c3c1

# Aggregating categories to general themes to avoid too many dummies
import re
# Create an aggregate sector variable which covers general topics
# For instance: "Transportation: Highways", "Transportation: Airports" -> "Transportation"
df.loc[:,'sector_aggregates'] = df['sector']
topics = ['Energy', 'Transportation']
for topic in topics:
    # Find all that contain the topic (ignore case), replace NaN with False (i.e., not found)
    # All found have same general topic
    df.loc[df['sector_aggregates'].str.contains(topic, re.IGNORECASE).replace(np.nan, False),'sector_aggregates'] = topic

# Outliers - Box plot: detect outliers that are outside the 1.5*IQR
# Keeping or removing them depends on our understanding of the data
# Try: boxplots, log transformation, scatterplot with target, Z score
sns.boxplot(x=df['variable'])
sns.boxplot(x=np.log(df['variable']))
df.plot.scatter(x='variable', y='price')
df['z_variable'] = stats.zscore(df['variable'])

# Outliers - Tukey rule
def tukey_filter(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR
    return df[(df[col_name] < max_value) & (df[col_name] > min_value)]

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

# Temporal data: Convert the dates to days since today
# Format: https://strftime.org
today = dt.datetime(2022,6,17)
for col in dat_cols:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')`# format='%d/%m/%Y', etc. 
    df[col] = df[col].apply(lambda col: int((today-col).days))

# Convert Unix time / timestamp to datetime
# 1381006850 -> 2013-10-05 21:00:50
import datetime
change_timestamp = lambda val: datetime.datetime.fromtimestamp(int(val)).strftime('%Y-%m-%d %H:%M:%S')
df['date'] = df['timestamp'].apply(change_timestamp)

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
# Use `np.triu()` if the matrix is very large to plot only one half
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt=".2f")
df.corr()['target'].sort_values(ascending=True).plot(kind='bar')
# -
correlations = df[selected_fields].corrwith(y) # correlations with target array y
correlations.sort_values(inplace=True)

# Similarity heatmaps
# If we have very large matrices (e.g., similarities between vectors)
# we might want to hide one half of the matrix (because it's symmetric)
# Taken from:
# https://github.com/mxagar/course_recommender_streamlit/blob/main/notebooks/02_FE.ipynb
sns.set_theme(style="white")
# Upper triangle of a matrix of ones: mask to hide upper half
mask = np.triu(np.ones_like(similarity_df, dtype=bool))
_, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Plot a similarity heat map
sns.heatmap(similarity_df, mask=mask, cmap=cmap, vmin=0.01, vmax=1, center=0, square=True)

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
sns.barplot(data=df, x='married', y='income', hue='gender')

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

### Word Cloud

# A word cloud can be a nice thing if we have a text field. Install:
# !pip install seaborn==0.11.1
# !pip install wordcloud==1.8.1

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Join all texts
texts = " ".join(title for title in df['text'].astype(str))

# English Stopwords: Defined in wordlcloud
stopwords = set(STOPWORDS)
# Custom stop words: common but uninteresting words from titles
stopwords.update(["getting started", "using", "enabling", "template", "university", "end", "introduction", "basic"])

# Configure
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400)

# Generate the wordcloud
wordcloud.generate(texts)

# Plot and save image
plt.axis("off")
plt.figure(figsize=(20,10))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
fig = plt.gca()
fig.get_xaxis().set_visible(False)
fig.get_yaxis().set_visible(False)
plt.savefig("./word_cloud.png", bbox_inches='tight')
plt.show()

### Plotly Express

# pip install plotly-express
import plotly.express as px

# TSNE reduction
tsne = TSNE(n_components=2,
            verbose=1,
            random_state=42)
z = tsne.fit_transform(X)

df_tsne = pd.DataFrame()
df_tsne["y"] = df["label"]
df_tsne["tsne-1"] = z[:,0]
df_tsne["tsne-2"] = z[:,1]
df_tsne["filename"] = df["filename"]

# Scatterplot: underlying filename shown when hovering on points
fig = px.scatter(df_tsne, x="tsne-1", y="tsne-2", color="y",
                 hover_data=["filename"])
fig.update_layout(
    width=1000, height=1000,
    title="Embedding vector T-SNE projection"
)
fig.show()

# Save interactive plot as HTML
# Points and their metadata saved in the HTML file! (so everything is there)
fig.write_html("interactive_scatter_plot.html")


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

# Ordinal encoder: there is a ranking order between category levels,
# we assume distances between categories are as encoded!
# handle_unknown: we use this flag in case new
categories = ['high school', 'bachelor', 'masters', 'phd']
oe = OrdinalEncoder(categories=categories, handle_unknown='ignore')
df['education_level'] = oe.fit_transform(df['education_level'])
# ['high school', 'bachelor', 'masters', 'phd'] -> [0, 1, 2, 3]
# NOTE: single columns might yield errors,
# in that case, we might need to reshape them:
ordinal_encoders = dict()
for col in categorical_features:
    oe = OrdinalEncoder()
    X[col] = oe.fit_transform(X[col].values.reshape(-1, 1))
    ordinal_encoders[col] = oe

# Make a feature explicitly categorical (as in R)
# This is not necessary, but can be helpful, e.g. for ints
# For strings of np.object, this should not be necessary
one_hot_int_cols = df.dtypes[df.dtypes == np.int].index.tolist()
for col in one_hot_int_cols:
    df[col] = pd.Categorical(df[col])

# One-hot encoding of features: Dummy variables with pandas
# Use drop_first=True to remove the first category and avoid multi-colinearity
# Note: if a field has NaN, all dummy variables will have a value of 0
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

### --- Natural Languange Processing (NLP): Extracting Text Features

## -- Create a vocabulary manually

from string import punctuation # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
from collections import Counter

text =  """This story is about surfing.
        Catching waves is fun!
        Surfing's a popular water sport.
        """

# Remove punctuation
for char in punctuation:
    text = text.replace(char, "")

# Count word appearances
# We can do that for each class/sentiment
# and then compute the ratios for each word!
word_counts = Counter()
for word in text.lower().split(): # split removing white spaces, \n, etc.
    word_counts[word] += 1

# Index -> Word (Vocabulary)
index2word = list(set(word_counts.keys()))

# Word -> Index
word2index = {}
for i,word in enumerate(index2word):
    word2index[word] = i

# Vocabulary size: number of words
n = len(index2word)

## -- Tokenize with NLTK

# More info:
# https://www.nltk.org/api/nltk.tokenize.html

# Built-in string split: it separates in white spaces by default
text = "Dr. Smith arrived late."
word = text.split() # ['Dr.', 'Smith', 'arrived', 'late.']

# NLTK: More meaningful word tokenization
from nlt.tokenize import word_tokenize
words = word_tokenize(text) # ['Dr.', 'Smith', 'arrived', 'late', '.']

# NLTK: Sentence splits or tokenization
from nlt.tokenize import sent_tokenize
text = "Dr. Smith arrived late. However, the conference hadn't started yet."
words = sent_tokenize(text)
# ['Dr. Smith arrived late.',
#  'However, the conference hadn't started yet.']

# SpaCy: More meaningful word tokenization
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
words = [token.text for token in doc]

## -- More with NLTK: Vocabulary, DTM/BoW, etc.

# With NLTK we can do many things, not only tokenization, e.g.:
# get PoS tag and filter according to it.
# The following snippet shows how to perform text feature extraction
# using NLTK.
# The example is from:
# https://github.com/mxagar/course_recommender_streamlit/blob/main/notebooks/02_FE.ipynb

import re
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim

# Download NLTK packages
# Note: NLTK sometimes needs to download the packages...
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
#nltk.download('maxent_ne_chunker')

# We tokenize the text colum values, BUT:
# - stop words are not considered
# - only nouns are taken; for that, we need to get the POS (part-of-speech) tags
def tokenize_text(text, keep_only_nouns=True):
    """Normalize and tokenize a text string.
    Stop words are removed.
    If specified, only nouns are taken.
    
    Args:
        text: str
            Text column/field
        keep_only_nouns: bool
            Whether to take only nouns or not (default: True)
    Returns:
        word_tokens: list[str]
            List of word tokens
    """
    stop_words = set(stopwords.words('english'))
    # Normalize: remove non-desired symbols, e.g., punctuation
    normalized_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    word_tokens = word_tokenize(normalized_text)
    # Remove English stop words and numbers
    word_tokens = [w for w in word_tokens
                   if (not w.lower() in stop_words) and (not w.isnumeric())]
    # Only keep nouns 
    if keep_only_nouns:
        # We can get a list of all POS tags with nltk.help.upenn_tagset()
        filter_list = ['WDT', 'WP', 'WRB', 'FW', 'IN', 'JJR', 'JJS',
                       'MD', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS',
                       'RP']
        tags = nltk.pos_tag(word_tokens)
        word_tokens = [word for word, pos in tags if pos not in filter_list]

    return word_tokens

# Run tokenization on the colum
# We get a list of lists: for each course a list of words/tokens
tokenized_texts = [tokenize_text(df.iloc[i,:]['text']) for i in range(df.shape[0])]
tokenized_texts[:1] # [['robots','coming',...]]

# Create a dictionary
tokens_dict = gensim.corpora.Dictionary(tokenized_texts)

# Vocalubary: token2id dictionary
print(tokens_dict.token2id) # {'ai': 0, 'apps': 1,...}

# Create Bags of Words (BoWs) for each tokenized course text
text_bows = [tokens_dict.doc2bow(text) for text in tokenized_texts]
text_bows[:1] # [[(0,2), (1,2), ...]]

# Create a dataframe which contains the BoWs of each text field
# flattened along the rows
doc_indices = [[df.iloc[i,:]['index']]*len(text_bows[i]) for i in range(df.shape[0])]
doc_ids = [[df.iloc[i,:]['text_id']]*len(text_bows[i]) for i in range(df.shape[0])]
tokens = [[tokens_dict.get(text_bows[i][j][0]) for j in range(len(text_bows[i]))] for i in range(len(text_bows))]
bow_values = [[text_bows[i][j][1] for j in range(len(text_bows[i]))] for i in range(len(text_bows))]

# Flatten the lists of lists
doc_indices = list(itertools.chain(*doc_indices))
doc_ids = list(itertools.chain(*doc_ids))
tokens = list(itertools.chain(*tokens))
bow_values = list(itertools.chain(*bow_values))

# Dictionary for the dataframe
bow_dicts = {"doc_index": doc_indices,
             "doc_id": doc_ids,
             "token": tokens,
             "bow": bow_values}

text_bows_df = pd.DataFrame(bow_dicts)

text_bows_df.head()
#       doc_index	doc_id	    token	bow
# 0	    0	        ML0201EN	ai	    2
# 1	    0	        ML0201EN	apps	2
# 2 	0	        ML0201EN	build	2
# 3	    0	        ML0201EN	cloud	1
# 4	    0	        ML0201EN	coming	1

## -- NLTK: Named Entity Recognition (NER)

import nltk
from nltk.tokenize import word_tokenize

nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

# Recognize named entities in a tagged sentence
# We need to first tokenize and POS-tag
text = "Antonio joined Udacity Inc. in California."
tree = ne_chunk(pos_tag(word_tokenize(text)))

# Display functions
print(tree)
tree.pretty_print()
tree.leaves()
tree.pprint()
for ne in tree:
    print(ne)

## -- NLTK: Stemming and Lemmatization

import nltk
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('omw-1.4')

from nltk.corpus import stopwords
text = "The first time you see The Second Renaissance it may look boring. Look at it at least twice and definitely watch part 2. It will change your view of the matrix. Are the human people the ones who started the war ? Is AI a bad thing ?"

# Normalize text
text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

# Tokenize text
words = text.split()
print(words)

# Stemming: modify endings
from nltk.stem.porter import PorterStemmer
stemmed = [PorterStemmer().stem(w) for w in words]
print(stemmed)

# Lemmatization: use dictionary + POS to find base form
from nltk.stem.wordnet import WordNetLemmatizer
# By default, in doubt, the lemma is a noun
lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
print(lemmed)
# Lemmatize verbs by specifying pos: when possible, lemma is a verb
lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
print(lemmed)

## -- NLP Processing with SpaCy

import spacy

# We load our English model
nlp = spacy.load('en_core_web_sm')

# Create a _Doc_ object:
# the nlp model processes the text 
# and saves it structured in the Doc object
# u: Unicode string (any symbol, from any language)
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

# Print each token separately
# Tokens are word representations, unique elements
# Note that spacy does a lot of identification work already
# $ is a symbol, U.S. is handled as a word, etc.
for token in doc:
    # token.text: raw text
    # token.pos_: part of speech: noun, verb, punctuation... (MORPHOLOGY)
    # token.dep_: syntactic dependency: subject, etc. (SYNTAX)
    # token.lemma_: lemma
    # token.is_stop: is the token a stop word?
    print(token.text, token.pos_, token.dep_)

# Loop in sentences
for sent in doc.sents:
    print(sent)

# Print the set of SpaCy's default stop words
print(nlp.Defaults.stop_words)

# Named entities (NER)
for ent in doc.ents:
    print(ent.text, ent.label_, str(spacy.explain(ent.label_)))

## -- CountVectorizer, TfidfVectorizer

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# We take only the text column
# Note that we don't need to generate an vocabulary manually
# this is done automatically by CountVectorizer or TfidfVectorizer!
# NOTE: `TfidfVectorizer` = `CountVectorizer` + `TfidfTransformer`,
# i.e., TfidfTransformer takes the ouput from CountVectorizer as input,
# whereas the TfidfVectorizer takes the same input as CountVectorizer

# URL regex
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# Custom text processing function
def preprocess_text(text):
    text = text.lower()
    # Remove all digits 
    text = re.sub(r'\d+', '', text)
    # Optionally: Remove punctuation characters:
    # Anything that isn't A through Z or 0 through 9 will be replaced with a space
    text = re.sub(r"[^a-zA-Z0-9", " ", text)
    # Optionally: remove URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    return text

# Create the DTM or frequency matrix with count values
# max_features: top features taken, according to their frequency in corpus
# preprocessor: we can pass any custom function we'd like
# We also can pass a tokenizer function if we want, that way X would be a list of texts.
cv = CountVectorizer(max_features = 500, preprocessor = preprocess_text)
cv_mat = cv.fit_transform(X)
# CountVectorizer.get_feature_names_out(): terms/tokens
cv_df = pd.DataFrame(cv_mat.toarray(), columns = cv.get_feature_names_out())
# 1586 rows × 500 columns

cv.vocabulary_ # term - index
cv.stop_words_ # words that were ignored due to several reasons...

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_mat = tfidf.fit_transform(X)
# Convert to a dataframe
# TfidfVectorizer normalizes each row to length 1
tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns = tfidf.get_feature_names_out())
# 1586 rows × 500 columns

tfidf.vocabulary_ # term - index
tfidf.idf_ # inverse document frequency vector
tfidf.stop_words_ # words that were ignored due to several reasons...


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
# If we want to plot it
component_variance_df = pd.DataFrame(data=cumsum.ravel(), columns=['variance'])
component_variance_df['components'] = range(1,features.shape[1]+1)
bplot = sns.barplot(data=component_variance_df, x='components', y = 'variance')
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

## Ranked Sum - Mann Whitney U

# Equivalent to Z/T-test, but without normality assumption
# https://github.com/mxagar/data_science_udacity/tree/main/04_ExperimentalDesign_RecSys/lab/Experiments

import scipy.stats as stats

def ranked_sum(x, y, alternative = 'two-sided'):
    """
    Return a p-value for a ranked-sum test, assuming no ties.
    
    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}
    
    Output value:
        p: estimated p-value of test
    """
    
    # compute U
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties
    
    # compute a z-score
    n_1 = x.shape[0]
    n_2 = y.shape[0]
    mean_u = n_1 * n_2 / 2
    sd_u = np.sqrt( n_1 * n_2 * (n_1 + n_2 + 1) / 12 )
    z = (u - mean_u) / sd_u
    
    # compute a p-value
    if alternative == 'two-sided':
        p = 2 * stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p = stats.norm.cdf(z)
    elif alternative == 'greater':
        p = stats.norm.cdf(-z)
    
    return p

## Sign Test

# Equivalent to Z/T-test with paired/repeated measures, but without normality assumption
# https://github.com/mxagar/data_science_udacity/tree/main/04_ExperimentalDesign_RecSys/lab/Experiments

import scipy.stats as stats

def sign_test(x, y, alternative = 'two-sided'):
    """
    Return a p-value for a ranked-sum test, assuming no ties.
    
    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}
    
    Output value:
        p: estimated p-value of test
    """
    
    # compute parameters
    n = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p = min(1, 2 * stats.binom(n, 0.5).cdf(min(k, n-k)))
    if alternative == 'less':
        p = stats.binom(n, 0.5).cdf(k)
    elif alternative == 'greater':
        p = stats.binom(n, 0.5).cdf(n-k)
    
    return p

## Experiment Size and Power

# Compute number of recordings/data-points necessary to see an effect of a given poser 1-beta
# assuming an alpha (Type I error).
# This formula works with ratios/proportions.
# https://github.com/mxagar/data_science_udacity/tree/main/04_ExperimentalDesign_RecSys/lab/Experiments

def experiment_size(p_null, p_alt, alpha = .05, beta = .20):
    """
    Compute the minimum number of samples needed to achieve a desired power
    level for a given effect size.
    
    WARNING: This formula is maybe one-sided, but we could use a two-sided approach?
    
    Input parameters:
        p_null: base success rate under null hypothesis
        p_alt : desired success rate to be detected
        alpha : Type-I error rate
        beta  : Type-II error rate
    
    Output value:
        n : Number of samples required for each group to obtain desired power
    """
    
    # Get necessary z-scores and standard deviations (@ 1 obs per group)
    z_null = stats.norm.ppf(1 - alpha)
    z_alt  = stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
    sd_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt) )
    
    # Compute and return minimum sample size
    p_diff = p_alt - p_null
    n = ((z_null*sd_null - z_alt*sd_alt) / p_diff) ** 2
    return np.ceil(n)

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
##### -- Data Modeling (Supervised Learning)
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
# Notes on the scoring (look also at online docu): 
# - can be a tring or a callable; accuracy is in general bad, prefer others: 'f1', 'roc_auc'
# - if multi-class, there are one-versus-rest versions, e.g. 'roc_auc_ovr'
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
# Notes on the scoring (look also at online docu): 
# - can be a tring or a callable; accuracy is in general bad, prefer others: 'f1', 'roc_auc'
# - if multi-class, there are one-versus-rest versions, e.g. 'roc_auc_ovr'
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
# Notes on the scoring (look also at online docu): 
# - can be a tring or a callable; accuracy is in general bad, prefer others: 'f1', 'roc_auc'
# - if multi-class, there are one-versus-rest versions, e.g. 'roc_auc_ovr'
search = GridSearchCV(estimator=SC, param_grid=param_grid, scoring='accuracy')
search.fit(X_train, y_train)
search.best_score_ # 1, be aware of the overfitting!
search.best_params_
# ---
#
# XGBoost
# Installation:
#   pip install graphviz
#   pip install xgboost
# Detailed guide:
#   https://github.com/mxagar/datacamp_notes/blob/main/Course_XGBoost/XGBoost_Datacamp.md

## XGBoost Classification - Example with Learning API
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

# Load Dataset
housing_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = housing_data[housing_data.columns.tolist()[:-1]]
y = housing_data[housing_data.columns.tolist()[-1]]
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size=0.2,
                                                   random_state=123)

# Create the DMatrix: DM_train, DM_test
# With the Learning API, we need to manually create the DMatrix data structure
DM_train = xgb.DMatrix(data=X_train,label=y_train)
DM_test =  xgb.DMatrix(data=X_test,label=y_test)

# Create the parameter dictionary: params
#   https://xgboost.readthedocs.io/en/stable/parameter.html
# Objective functions:
#   reg:linear - regression (deprecated)
#   reg:squarederror - regression
#   reg:logistic - classification, class label output
#   binary:logistic - classification, class probability output
params = {"objective":"reg:logistic",
          "booster":"gbtree", # Trees are "gbtree" (default), Linear learners (avoid) "gblinear"
          "max_depth":2}
# Train: same API for regression/classification
# The type of problem is defined in param["objective"]
xgbm = xgb.train(params=params,
                 dtrain=DM_train,
                 num_boost_round=10) # = number of weak learners
# Predict
preds = xgbm.predict(DM_test)

# Alternative: Cross-Validation
# Perform cross-validation with another metric: AUC
cv_results = xgb.cv(dtrain=DM_train,
                    params=params, 
                    nfold=3, # number of non-overlapping folds
                    num_boost_round=5, # number of base learners
                    metrics="auc", # "error", "rmse", "mae", ...
                    as_pandas=True, # If matrix should be converted to pandas
                    seed=123)
# Print cv_results
print(cv_results)
#    train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
# 0        0.907307       0.025788       0.694683      0.057410
# 1        0.951466       0.017800       0.720245      0.032604
# 2        0.975673       0.009259       0.722732      0.018837


## XGBoost Regression - Example with Scikit-Learn API and RandomizedSearchCV
# The Scikit-Learn API is used via xgb.XGBRegressor() or gb.XGBClassifier()
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# Load Dataset
housing_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X = housing_data[housing_data.columns.tolist()[:-1]]
y = housing_data[housing_data.columns.tolist()[-1]]
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size=0.2,
                                                   random_state=123)
# Parameter space tested by RandomizedSearchCV
# All possible combinations are: 20 * 1 * 20 = 400
# BUT: we limit to n_iter=25 the number of combinations
# And we will train each of them 4-fold with CV
# NOTE: The parameter names with the Scikit-Learn API are different
# eta -> learning_rate
# num_boost_round -> n_estimators
gbm_param_grid = {'learning_rate': np.arange(0.05,1.05,.05), # arange: 20 values
                  'n_estimators': [200],
                  'subsample': np.arange(0.05,1.05,.05)} # arange: 20 values

# Scikit-Learn API
# We can fit() and predict():
#   gbm.fit(X_train, y_train)
#   preds = gbm.predict(X_test)
gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm,
                                    param_distributions=gbm_param_grid,
                                    n_iter=25, # number of combinations
                                    scoring='neg_mean_squared_error',
                                    cv=4,
                                    verbose=1)

randomized_mse.fit(X_train, y_train)
print("Best parameters found: ",randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

## XGBoost: Tree and Feature Importance Visualization
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

housing_data = pd.read_csv("../data/ames_housing_trimmed_processed.csv")
X,y = housing_data.iloc[:,:-1],housing_data.iloc[:,-1]
housing_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective":"reg:squarederror", "max_depth":2}
xg_reg = xgb.train(params=params,
                   dtrain=housing_dmatrix,
                   num_boost_round=10) # 10 trees in total

# Plot the first tree
# num_trees refers to the tree, starting with 0
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()

#
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
# Another way of plotting the confusion matrix
# If we have used the LabelEncoder(), we can retrieve the class names
# and pass them as parameters: display_labels=le.classes_,
fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(
    model,
    X_test,
    y_test,
    ax=sub_cm,
    normalize="true",
    values_format=".1f",
    xticks_rotation=90,
)
fig_cm.tight_layout()

# ROC-AUC scores can be calculated by binarizing the data
# label_binarize performs a one-hot encoding,
# so from an integer class we get an array of one 1 and the rest 0s.
# This is necessary for computing the ROC curve, since the target needs to be binary!
# Again, to get a single ROC-AUC from the 6 classes, we pass average='weighted'
auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
          label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
          average='weighted')
# Old interface, not available anymore:
# model_roc_plot = plot_roc_curve(model, X_test, y_test, name="Logistic Regression") # ROC curve plotted and AUC computed
# An alternative is the following;
# note that
# - we need to use y_prob = model.predict_proba(X_test)
# - we need to plot manually
def plot_roc_curve(model, X_test, y_test, title, filename):
    """Plot ROC curve."""
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=None)
    plt.figure(figsize=(5,5))
    plt.title(f'Receiver Operating Characteristic - ROC\n({title})')
    plt.plot(fpr, tpr, label=f"Model predictions, AUC (test) = {round(auc,2)}")
    plt.plot([0, 1], ls="--", label="Random guess, AUC = 0.5")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig(f'./assets/{filename}', dpi=200, transparent=False, bbox_inches='tight')

title = "subtitle"
filename = "model_roc_curve_test.png"
plot_roc_curve(model, X_test, y_test, title, filename)

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
def plot_real_vs_predicted(model, X_test, y_test, variable_name, title, filename):
    """Plot real y vs. predicted."""
    plt.figure(figsize=(6,6))
    x = y_test
    y = model.predict(X_test)
    r2_test = r2_score(x,y)
    plt.scatter(x, y, color='b', alpha=1.0)
    plt.plot(np.array([0,np.max(x)],dtype='object'),np.array([0,np.max(x)],dtype='object'),'r-')
    plt.legend([f'{title}, R2 = {r2_test}','Ideal: Predicton = True'])
    plt.xlabel(f'True {variable_name}')
    plt.ylabel(f'Predicted {variable_name}')
    plt.title(f'Evaluation of Model Predictions\n({title})')
    plt.axis('equal')
    plt.savefig(f'./assets/{filename}',dpi=600,transparent=True,bbox_inches='tight')

variable_name = "variable"
title = "subtitle"
filename = "model_prediction_performance.png"
plot_real_vs_predicted(model, X_test, y_test, variable_name, title, filename)    
    
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
importance.index = df.columns # Or, better: model.feature_names_in_
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
from sklearn.inspection import permutation_importance
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

### Feature Importances with SHAP

# SHAP library for model explanation
# https://github.com/slundberg/shap
# pip install shap
import shap

# Build Model
model = RandomForestRegressor()
model.fit(X, y)

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")

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
# Notes on the scoring (look also at online docu): 
# - can be a tring or a callable; accuracy is in general bad, prefer others: 'f1', 'roc_auc'
# - if multi-class, there are one-versus-rest versions, e.g. 'roc_auc_ovr'
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

### --- Hierarchical Pipelines: ColumnTransformer + make_pipeline + GridSearchCV

import itertools
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

# Define treated columns
categorical_features = ['a', 'b', 'c']
numerical_features = ['one', 'two']
nlp_features = ['review']

# Define processing for categorical columns
# handle_unknown: label encoders need to be able to deal with unknown labesl!
categorical_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0),
    OrdinalEncoder(handle_unknown='ignore')
)
# We can use make_pipeline if we don't care about accessing steps later on
# but if we want to access steps, better to use Pipeline!
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])

# Define processing for numerical columns
numerical_transformer = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler()
)
# Define processing of NLP/text columns
# This trick is needed because SimpleImputer wants a 2d input, but
# TfidfVectorizer wants a 1d input. So we reshape in between the two steps
# WARNING: If we use lambdas as part off pipeline elements, these might have
# problems while being serialized as pickles; instead, use function definitions
# which can be imported and/or skops as serializer!
reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
nlp_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=""),
    reshape_to_1d,
    TfidfVectorizer(
        binary=True,
        max_features=10
    )
)

# Put the 3 tracks together into one pipeline using the ColumnTransformer
# This also drops the columns that we are not explicitly transforming
processor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
        ("nlp1", nlp_transformer, nlp_features),
    ],
    remainder="drop",  # This drops the columns that we do not transform
)

# Get a list of the columns we used
used_columns = list(itertools.chain.from_iterable([x[2] for x in processor.transformers]))
# BUT: This is not the final set of feature names!
# If we use OneHotEncoder or similar transformers, 
# https://stackoverflow.com/questions/54646709/sklearn-pipeline-get-feature-names-after-onehotencode-in-columntransformer
# https://stackoverflow.com/questions/54570947/feature-names-from-onehotencoder
# AFTER WE HAVE FIT processor, we can get the dummy column names.
# For instance, the dummy categorical names:
cat_names = list(processor.transformers_[1][1]\
    .named_steps['onehot'].get_feature_names_out(categorical_features))
# Remove blank spaces
cat_names = [col.replace(' ', '') for col in cat_names]

# In production, avoid leaving default parameters to models,
# because defaults can change.
# Instead, save parameters in YAML files and load them as dicts;
# we can pass them to models at instantiation! 
# Of course, dict key-values must be as defined in the model class.
config = dict()
with open('model_configuration.yaml') as f:
    config = yaml.safe_load(f)

# Append classifier to processing pipeline.
# Now we have a full prediction pipeline.
# Pipeline needs to be used here.
# The result is a pipeline with two high-level elements: processor and classifier.
# Note that we pass the configuration dictionary to the model;
# however, this should be modified in the grid search.
pipe = Pipeline(
    steps=[
        ("processor", processor),
        ("classifier", RandomForestClassifier(**config)),
    ]
)

# Define Grid Search: parameters to try, cross-validation size
param_grid = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None]+[n for n in range(5,20,5)]
}
# Grid search
search = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      cv=3,
                      scoring='roc_auc_ovr') # ovr = one versus rest, to make roc_auc work with multi-class
# Find best hyperparameters and best estimator pipeline
search.fit(X, y)
rfc_pipe = search.best_estimator_

print(search.best_score_)
print(search.best_params_)

### --- Other (More Sophisticated) Pipeline Elements

# Example project:
# https://github.com/mxagar/disaster_response_pipeline/
#
# MultiOutputClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
estimator = MultiOutputClassifier(RandomForestClassifier())
estimator.fit(X, y)

# Feature Union
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD
union = FeatureUnion([("pca", PCA(n_components=1)),
                      ("svd", TruncatedSVD(n_components=2))])
X = [[0., 1., 3], [2., 2., 5]]
union.fit_transform(X)

### -- 
### -- Dataset Structure (Unsupervised Learning)
### -- 

### --- Clustering Commands Applied with K-Means

# Alternative with mini-batches: MiniBatchKMeans
from sklearn.cluster import KMeans

# Parameter: n_clusters
# 'k-means++' initializes first centroids far from each other
kmeans = KMeans(n_clusters=3,
                init='k-means++')
# Fit dataset
kmeans.fit(X1)
# We can predict the clusters of another dataset!
y_pred = kmeans.predict(X2)

# Inertia: inertia_k = sum(i = 1:n; (x_i - C_k)^2)
kmeans.inertia_
# Cluster centers: C_k
km.cluster_centers_
# Labels: C_k assigned to each data point
km.labels_

# Elbow method: fit clusterings with different number of centroids
# and take the one from which metric (e.g., inertia) doesn't improve significantly
inertia = []
clusters = list(range(2,10))
for k in clusters:
    kmeans = KMeans(n_clusters=k,
                    init='k-means++',
                    random_state=10) # always define it!
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)
# Plot k vs inertia
plt.plot(clusters,inertia)
plt.scatter(clusters,inertia)
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia');

# Plot clustered data in 2D
def plot_dataset_clusters(X,km=[],num_clusters=0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1], c=color[0], alpha=alpha, s=s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0], X[km.labels_==i,1], c=color[i], alpha=alpha, s=s)
            plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], c=color[i], marker = 'x', s = 100)
    plt.show()

### --- Other Clustering Algorithms

## Hierarchical Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering

# We can decide the number of clusters or a distance threshold as criterium:
# distance_threshold, n_clusters
# Linkage:
# 'single': distance between closest points in different clusters
# 'complete': compute the furthest point pairs in different clusters and select the minimum pair
# 'average': distances between cluster centroids
# 'ward': pair which minimizes the inertia is merged; this is similar to K-means
agg = AgglomerativeClustering(  n_clusters=3, 
                                affinity='euclidean', # distance metric
                                linkage='ward')
agg.fit(X1)
y_pred = agg.labels_

## DBSCAN

from sklearn.cluster import DBSCAN

# Parameters:
# - Distance metric.
# - epsilon: radius of local neighborhood
# - min_samples: number of samples in neighborhood for a point to be considered as a core poin
db = DBSCAN(eps=3,
            metric='euclidean',
            min_samples=3)
db.fit(X)
# You cannot call predict,
# instead, you get the clusters for the current dataset 
# labels: -1, 0, 1, 2, ...
# Noisy samples are given the label -1
clusters = db.labels_

## Mean Shift

from sklearn.cluster import MeanShift, estimate_bandwidth

# Estimate the bandwidth, parameters:
# - X: (n_samples, n_features)
# - quantile: float, default=0.3 Should be between [0, 1]; 0.5 = median of all pairwise distances used.
# - n_samples: int, number of samples to be used; if not given, all samples used.
bandwidth = estimate_bandwidth(X1, quantile=.06, n_samples=3000)

# Mean Shift, parameters:
# - max_itert: (default=300) maximum number of iterations per seed point, if not converged.
# - bin_seeding :if True, initial kernel locations are not locations of all points, but rather the location of the discretized version of points.
ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X1)

# Manual bandwidth
ms = MeanShift(bandwidth=2)
ms.fit(X1)
clusters = ms.predict(X2)

# Get labels for each data point
labels = ms.labels_
# Get all unique clusters
np.unique(labeled) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# Get cluster centers
ms.cluster_centers_ # (12, 3) array


### --- Clustering for Image Compression

img = plt.imread('peppers.jpg', format='jpeg')
plt.imshow(img)
plt.axis('off')

# Each pixel with its [R,G,B] values becomes a row
# -1 = img.shape[0]*img.shape[1], because we leave the 3 channels
# as the second dimension
img_flat = img.reshape(-1, 3)

img.shape # (480, 640, 3)
img_flat.shape # (307200, 3)

# Note that in reality we have 256^3 possible colors = 16,777,216
# but not all of them are used.
# All the unique/used colors
len(np.unique(img_flat,axis=0)) # 98452

# K=8 clusters: we allow 8 colors
kmeans = KMeans(n_clusters=8, random_state=0).fit(img_flat)

# Loop for each cluster center
# Assign to all pixels with the cluster label
# the color of the cluster == the cluster centroid
img_flat2 = img_flat.copy()
for i in np.unique(kmeans.labels_):
    img_flat2[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]

img2 = img_flat2.reshape(img.shape)
plt.imshow(img2)
plt.axis('off');

### --- Clustering / Anomaly Detection: Gaussian Mixtures Model

from sklearn.mixture import GaussianMixture

# covariance_type
# full: each component has its own general covariance matrix.
# tied: all components share the same general covariance matrix.
# diag: each component has its own diagonal covariance matrix.
# spherical: each component has its own single variance.
gmm = GaussianMixture(n_components=3,
                      covariance_type='tied',
                      init_params='kmeans')
gmm.fit(X1)

# We can get labels or another dataset
labels = gmm.predict(X2)
# ... or probabilities!
# Probabilities make possible a "soft" cluster choice
# or ANOMALY DETECTION: if data point has low probs
# for all clusters, it's an outlier
probs = GMM.predict_proba(X2)

# Extract model parameters
gmm.weights_ # weights of each mixture components
gmm.means_ # (n_components, n_features)
gmm.covariances_ # spread, size depends on covariance_type

### --- Clustering as Feature Engineering

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Does it the number of clusters have an effect? Steps to see that:
# Create 10 strata
# Create the basis training set from by taking float_columns.
# For n = 1..20, fit a KMeans algorithim with n clusters.
# Add data point cluster as feature, and one-hot-encode it.
# Fit 10 Logistic Regression models (one for each stratum)
# and compute the average roc-auc-score.
# Plot the average roc-auc scores.

# Create 10 strata
sss = StratifiedShuffleSplit(n_splits=10, random_state=6532)

# Given X, y, and an estimator
# Compute the average ROC in the 10 stratifications
def get_avg_roc_10splits(estimator, X, y):
    roc_auc_list = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        estimator.fit(X_train, y_train)
        y_predicted = estimator.predict(X_test)
        y_scored = estimator.predict_proba(X_test)[:, 1]
        roc_auc_list.append(roc_auc_score(y_test, y_scored))
    return np.mean(roc_auc_list)

# Apply K-means with n clusters
# Then one-hot-encode them
def create_kmeans_columns(n):
    km = KMeans(n_clusters=n)
    km.fit(X_basis)
    km_col = pd.Series(km.predict(X_basis))
    km_cols = pd.get_dummies(km_col, prefix='kmeans_cluster')
    return pd.concat([X_basis, km_cols], axis=1)

# New target: good quality wine vs not good quality wine
y = (data['quality'] > 7).astype(int)
# Basis X: all float collumns without kmeans cluster ids
X_basis = df[float_columns]

# Create 10 LR models for each number of clusters n
# Get average ROC for each n
estimator = LogisticRegression()
ns = range(1, 21)
roc_auc_list = [get_avg_roc_10splits(estimator, create_kmeans_columns(n), y)
                for n in ns]

# Plot: n vs average ROC
ax = plt.axes()
ax.plot(ns, roc_auc_list)
ax.set(
    xticklabels= ns,
    xlabel='Number of clusters as features',
    ylabel='Average ROC-AUC over 10 iterations',
    title='KMeans + LogisticRegression'
)
ax.grid(True)

### --- Distance metrics

from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import (cosine_distances,
                                      paired_euclidean_distances,
                                      paired_manhattan_distances,
                                      cosine_similarity)

# This function will allow us to find the average distance between two sets of data
def avg_distance(X1, X2, distance_func):
    from sklearn.metrics import jaccard_score
    #print(distance_func)
    res = 0
    for x1 in X1:
        for x2 in X2:
            if distance_func == jaccard_score: # the jaccard_score function only returns jaccard_similarity
                res += 1 - distance_func(x1, x2)
            else:
                res += distance_func(x1, x2)
    return res / (len(X1) * len(X2))


# Jaccard distance for categorical datasets
df.columns # All categorical, even age
# 'Class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
# 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'

print(sorted(df['age'].unique()))
# ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

# One-hot encode all columns except age
from sklearn.preprocessing import OneHotEncoder
OH = OneHotEncoder()
X = OH.fit_transform(df.loc[:, df.columns != 'age']).toarray()

# Take two strata: two age groups
X30to39 = X[df[df.age == '30-39'].index]
X60to69 = X[df[df.age == '60-69'].index]
X30to39.shape, X60to69.shape
# ((36, 39), (57, 39))

avg_distance(X30to39, X30to39, jaccard_score)
# 0.6435631883548536

## Distances with scipy
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
from scipy.spatial.distance import cosine, jaccard, euclidean, cityblock
bow_1 = np.array([1, 1, 0, 1, 1])
bow_2 = np.array([1, 1, 1, 0, 1])
dist_funcs = [cosine, jaccard, euclidean, cityblock]
for f in dist_funcs:
    d = f(bow_1, bow_2)
    if f == cosine:
        # similarity = 1 - cosine
        d = 1 - d
    print(d)

### --- Dimensionality Reduction: PCA

from sklearn.decomposition import PCA

# Imagine our dataset has n=20 features and m>n samples
# and we want to reduce it to k=3 features; we apply PCA/SVD:
# X_(mxn) = U_(mxm) * S_(mxn) * V^T_(nxn)
# X_hat_(mxn) = U_(mxk) * S_(kxk) * V^T_(kxn)
# IMPORTANT: X must be scaled, or all features in similar ranges!
pca = PCA(n_components=3) # final number of components we want
X_hat = pca.fit_transform(X)

# We can get many information from pca
# Principal axes in feature space: V^T, (n_components, features=X.shape[1])
pca.components_
# Variance ratio of each component
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum() # must be 1
# Variance of each component
pca.explained_variance_
# Sum all explained variances until 95% is reached;
# how many components do we need?
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >=0.95) + 1
# If we want to plot it:
component_variance_df = pd.DataFrame(data=cumsum.ravel(), columns=['variance'])
component_variance_df['components'] = range(1,features.shape[1]+1)
bplot = sns.barplot(data=component_variance_df, x='components', y = 'variance')

### --- Dimensionality Reduction: Kernel PCA with GridSearchCV

# In this example we use KernelPCA,
# which is a non-linear dimensionality reduction
# through the use of kernels.
# The goal is to show how we can perform grid search to find the best
# parameters.
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Custom scorer:
# We want to find the best parameters for kernel PCA
# We can use GridSearchCV,
# but a custom scoring function needs to be defined.
# The score is the opposite of the error,
# so we compute the error between the original and the
# inverse transformed dataset
# and use its negative value
def scorer(pcamodel, X, y=None):

    try:
        X_val = X.values
    except:
        X_val = X
        
    # Calculate and inverse transform the data
    data_inv = pcamodel.fit(X_val).transform(X_val)
    data_inv = pcamodel.inverse_transform(data_inv)
    
    # The error calculation
    mse = mean_squared_error(data_inv.ravel(), X_val.ravel())
    
    # Larger values are better for scorers, so take negative value
    return -1.0 * mse

# The grid search parameters
# kernel: 'rbf' = Gaussian
# gamma: parameter for the Guassian (sigma), how complex/curvy the boundary should be
param_grid = {'gamma':[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
              'n_components': [2, 3, 4]}

# The grid search
kernelPCA = GridSearchCV(KernelPCA(kernel='rbf', fit_inverse_transform=True),
                         param_grid=param_grid,
                         scoring=scorer,
                         n_jobs=-1)
kernelPCA = kernelPCA.fit(X)

# Best estimator
kernelPCA.best_estimator_

### --- Dimensionality Reduction: MDS, TSNE

from sklearn.manifold import MDS

# Create an MDS embedding
# n_componenets: dimension of reduced embedding
# dissimilarity: 'euclidean' (default) or 'precomputed'
#   this is a very important argument
#   if 'euclidean' (default), we pass a dataset of points
#   and pairwise Euclidean distances between them are computed
#   if 'precomputed', MSD expects a matrix of precomputed distances
#   not data points; we can use any metric to compute those distances!
embedding=MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2)

# Transform the digits to the embedding
X_transformed=embedding.fit_transform(X)
# Now, plot the embedding: scatter and color according to group
# See K-means examples

# Now we pass the dissimilarity matrix instead of the initial dataset
# and we use our distance metric of choice
# Different embeddings are created.
# As in the case of the cities, it is remarkable that we don't pass the initial dataset, but the matrix of distances,
# and it works!
from scipy.spatial.distance import squareform, pdist

dist=['cosine','cityblock','hamming','jaccard','chebyshev','canberra','braycurtis']
scaler = MinMaxScaler()
X_norm=scaler.fit_transform(X)

for d in dist:
    # distance is a n_sample x n_sample matrix with distances between samples
    distance=squareform(pdist(X_norm,metric=d))
    print( d)

    embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2)
    X_transformed = embedding.fit_transform(distance)
    # Now, we could plot the embedding

# TSNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, init='random').fit_transform(X)
fig, ax = plt.subplots()
# Now, we could plot the embedding

### --- Dimensionality Reduction: NMF

# This is usually used with
# - reductions that need a higher interpretablity
# - topic discovery
# - recommender systems: user-item matrix factorization

## NMF Topic Discovery

from sklearn.decomposition import NMF

# n_components: latent components to be identified, e.g., topics
# X: documents x words
nmf = NMF(n_components=5, init='random', random_state=818)
# The transformed data points: they are represented in the new basis,
# for each data point, we get the positive weight of each basis component
W = nmf.fit_transform(X) # (X.shape[0]=n_samples, n_components)
# The new basis with n_components intepretable vectors
# For each topic, the importance of each word
# or vice versa: for each word, their important in each topic
H = nmf.components_ # (n_components, X.shape[1]=n_features)
# X_hat = V = W@H

# Find feature with highest topic-value per doc
np.argmax(W, axis=1)

# Analysis
# The real words should be extractable from X,
# if it has been created with CountVectorizer or TfidfVectorizer
# words <- countvectorizer.vocabulary_
words = list(range(1,X.shape[0]+1))
documents = list(range(1,X.shape[1]+1))

topic_word = pd.DataFrame(data=H.round(3),
                         columns=words,
                         index=['topic_1','topic_2','topic_3','topic_4','topic_5'])
topic_doc = pd.DataFrame(data=W.round(3),
                         index=documents,
                         columns=['topic_1','topic_2','topic_3','topic_4','topic_5'])
# The mapping from topic_x to the topic label
topic_doc.reset_index().groupby('index').mean().idxmax()
# topic_1    3
# topic_2    4
# topic_3    1
# topic_4    5
# topic_5    2

# The most important 20 words for topic x
topic_word.T.sort_values(by='topic_5', ascending=False).head(20)

## NMF Recommender System: User-course ratings
# Origin: https://github.com/mxagar/course_recommender_streamlit/blob/main/notebooks/04_Collaborative_RecSys.ipynb
# Note: Usually the library Suprise works better (also in the link above)

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dense ratings
ratings_df = pd.read_csv("../data/ratings.csv")
#	user	item	rating
# 0	1889878	CC0101EN	3.0
# 1	1342067	CL0101EN	3.0
# ...
# Dense -> Sparse ratings
# WARNING: make sure if you really need fillna() or not
ratings_sparse_df = ratings_df.pivot(index='user', columns='item', values='rating').fillna(0).reset_index().rename_axis(index=None, columns=None)
#   user	AI0111EN	BC0101EN	...
# 0	2	    0.0	        2.0
# ...
# Pivot might very memory expensive, 
# if we have issues with it, an alternative could be:
# https://stackoverflow.com/questions/39648991/pandas-dataframe-pivot-not-fitting-in-memory
# WARNING: no fillna() is applied in this example, in contrast to before
ratings_sparse_df = ratings_df.groupby(['user', 'item'])['rating'].max().unstack()
# Imagine we have a user-item interaction dataframe without ratings, just interactions
# We can create a temp column with interactions and from it create a 0/1 interactions matrix as follows 
# df.columns: article_id, title, user_id
df['interaction'] = 1
intractions_df = df.groupby(['user_id', 'article_id'])['interaction'].max().unstack().fillna(0)

# Scikit-learn uses the SPARSE representation
X_train, X_test = train_test_split(
    ratings_sparse_df.iloc[:,1:],
    test_size=0.3, # portion of dataset to allocate to test set
    random_state=42 # we are setting the seed here, ALWAYS DO IT!
)

nmf = NMF(n_components=15, init='random', random_state=818)
W = nmf.fit_transform(X_train) # (X.shape[0]=n_samples, n_components)
H = nmf.components_ # (n_components, X.shape[1]=n_features)
X_hat = W@H

W.shape # (23730, 15)
H.shape # (15, 126)

print('RMSE: ', mean_squared_error(X_train, X_hat, squared=False))
# RMSE:  0.3360953022513968

# IMPORTANT: The fitted NMF model has constant H components,
# but W is different for each input data X.
W_test = nmf.transform(X_test)
W_test.shape # (10171, 15)

X_test_hat = W_test@H

print('RMSE: ', mean_squared_error(X_test, X_test_hat, squared=False))
# RMSE:  0.3373680254871046

