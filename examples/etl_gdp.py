"""This module contains an ETL pipeline
which processes the GDP data obtained from the
World Bank website.

ETL pipelines (i.e., Extract, Transform, Load)
are common before applying any data analysis or
modeling. In the present case, these steps are
carried out, mainly executed by the function run_etl():

- A source file stream is opened (DATA_SOURCE_FILENAME)
- It is read it line by line
- Each line is cleaned and transformed by transform_indicator_data()
- Each transformed line is inserted the output to the goal database
    with load_indicator_data()

To use this file, check that DATA_SOURCE_FILENAME points
to the correct file path and run the script:

    $ python etl_gdp.py

Then, you should get evidence/output in the terminal
and the SQLite database DB_FILENAME should be there.

Source: Udacity Data Science Nanodegree exercise,
link to original file:

    https://github.com/mxagar/data_science_udacity/blob/main/03_DataEngineering/lab/18_final_exercise/etl_gdp.py

Author: Mikel Sagardia
Date: 2023-03-06
"""
import sqlite3
import numpy as np
import pandas as pd

DATA_SOURCE_FILENAME = "./data/gdp_data.csv"
DB_FILENAME = "./data/world_bank.db"
DB_TABLE_NAME = "gdp"

def create_database_table():
    """Create the database file with the gdp table.
    
    Args: None
    Returns: None
    """
    # Connect to the database
    # sqlite3 will create this database file if it does not exist already
    conn = sqlite3.connect(DB_FILENAME)

    # Get a cursor
    cur = conn.cursor()

    # Drop the gdp table in case it already exists
    cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_NAME}")

    # Create the gdp table: long format, with these rows:
    # countryname, countrycode, year, gdp
    sql_string = f"""CREATE TABLE {DB_TABLE_NAME} (countryname TEXT,
                                                   countrycode TEXT,
                                                   year INTEGER,
                                                   gdp REAL,
                                                   PRIMARY KEY (countrycode, year));"""
    cur.execute(sql_string)

    # Commit and close
    conn.commit()
    conn.close()

def extract_lines(file):
    """Generator for reading one line at a time.
    Generators are useful for data sets that are too large
    to fit in RAM.
    
    Usage:
    
        with open('dataset.csv') as f:
            for line in extract_lines(f):
                data = line.split(',')
                # process row/line...

    Args:
        file (object): file opened with open()

    Returns:
        line (str): yield one line string
    """
    while True:
        line = file.readline()
        if not line:
            break
        yield line

def transform_indicator_data(data, col_names):
    """A single line of a data source is cleaned and transformed
    into a usable format.

    The argument data is a list which represents a row
    from a dataframe with the following 63 col_names (in order):
    
    - 'Country Name': not only countries, also continents, etc.
    - 'Country Code'
    - 'Indicator Name': 'GDP (current US$)' for all
    - 'Indicator Code': 'NY.GDP.MKTP.CD' for all
    - 1960
    - 1961
    - ...
    - 2016
    - 2017
    - 'Unnamed: 62': NaN for all

    That list is converted to long format, which results in a
    dataframe with the following columns:
    
    - countryname
    - countrycode
    - year
    - gdp

    In the process:
    - Unnecessary columns are dropped
    - Only real countries are taken (i.e., not continents)
    - Only GDP values which are not NaN are taken
    
    The return is a list of lists of the following form:
    
        [[countryname, countrycode, year, gdp]]

    Args:
        data (list): data point of a country, as described above.
        col_names (list): column names of the data point

    Returns:
        results (list): GDP values per year & country, as described above.
    """
    # Get rid of quote marks
    for i, datum in enumerate(data):
        data[i] = datum.replace('"','')
    
    # Extract country name
    country = data[0]
    
    # These are "countryname" values that are not actually countries
    # List generated from visual inspection
    non_countries = ['World',
     'High income',
     'OECD members',
     'Post-demographic dividend',
     'IDA & IBRD total',
     'Low & middle income',
     'Middle income',
     'IBRD only',
     'East Asia & Pacific',
     'Europe & Central Asia',
     'North America',
     'Upper middle income',
     'Late-demographic dividend',
     'European Union',
     'East Asia & Pacific (excluding high income)',
     'East Asia & Pacific (IDA & IBRD countries)',
     'Euro area',
     'Early-demographic dividend',
     'Lower middle income',
     'Latin America & Caribbean',
     'Latin America & the Caribbean (IDA & IBRD countries)',
     'Latin America & Caribbean (excluding high income)',
     'Europe & Central Asia (IDA & IBRD countries)',
     'Middle East & North Africa',
     'Europe & Central Asia (excluding high income)',
     'South Asia (IDA & IBRD)',
     'South Asia',
     'Arab World',
     'IDA total',
     'Sub-Saharan Africa',
     'Sub-Saharan Africa (IDA & IBRD countries)',
     'Sub-Saharan Africa (excluding high income)',
     'Middle East & North Africa (excluding high income)',
     'Middle East & North Africa (IDA & IBRD countries)',
     'Central Europe and the Baltics',
     'Pre-demographic dividend',
     'IDA only',
     'Least developed countries: UN classification',
     'IDA blend',
     'Fragile and conflict affected situations',
     'Heavily indebted poor countries (HIPC)',
     'Low income',
     'Small states',
     'Other small states',
     'Not classified',
     'Caribbean small states',
     'Pacific island small states']
    
    # Filter out country name values that are in the above list
    if country not in non_countries:        
        # Convert the data variable into a numpy array
        data_array = np.array(data, ndmin=2)
        
        # Reshape the data_array so that it is one row and 63 columns
        data_array.reshape(1,63)
        
        # Convert the data_array variable into a pandas dataframe
        # Specify the column names as well using the col_names variable/arg
        # Replace all empty strings in the dataframe with np.nan
        df = pd.DataFrame(data_array, columns=col_names).replace('', np.nan)
        
        # Drop the unnecessary columns
        #df.drop(['\n', 'Indicator Name', 'Indicator Code', 'Unnamed: 62'], inplace=True, axis=1)
        df.drop(['\n', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)
        # Reshape the data sets so that they are in long format
        #   Country Name, Country Code year, gdp
        df_melt = df.melt(id_vars=['Country Name', 'Country Code'], 
                            var_name='year', 
                            value_name='gdp')
        
        # For each row in df_melt, extract the following values into a list:
        #   [country, countrycode, year, gdp]
        results = []
        for index, row in df_melt.iterrows():
            country, countrycode, year, gdp = row
            # Check if gpd is NaN: convert it to str
            if str(gdp) != 'nan':
                results.append([country, countrycode, year, gdp])
                
        return results

def load_indicator_data(results):
    """Insert the GDP data per country and year transformed
    by transform_indicator_data() into the goal database.

    The input results should have the following form:
    
        [[countryname, countrycode, year, gdp], [...], ...]

    Args:
        results (list): list of lists returned by transform_indicator_data()
            which contains GDP data per country and year

    Returns: None
    """
    # Connect to the goal database using the sqlite3 library
    conn = sqlite3.connect(DB_FILENAME)
    
    # Create a cursor object
    cur = conn.cursor()
    
    if results: 
        for result in results:
            # Extract the values from each list in the big results list
            countryname, countrycode, year, gdp = result

            # Prepare a query to insert a countryname, countrycode, year, gdp value
            sql_string = f"""INSERT INTO {DB_TABLE_NAME} 
                             (countryname, countrycode, year, gdp)
                             VALUES ("{countryname}", "{countrycode}", {year}, {gdp});"""

            # Connect to database and execute query
            try:
                cur.execute(sql_string)
            except Exception as e:
                print('error occurred:', e, result)
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()

def run_etl():
    """Run the ETL pipeline: Extract, Transform, Load.
    This function
    
    - opens a file stream
    - read it line by line
    - transforms each line with transform_indicator_data()
    - inserts the output to the goal database with load_indicator_data()
    
    Args: None
    Returns: None
    """
    # Create database table
    create_database_table()
    
    # ETL
    with open(DATA_SOURCE_FILENAME) as f:
        # Execute the generator to read in the file line by line
        for line in extract_lines(f):
            # Split the comma separated values
            data = line.split(',')
            # Check the length of the line because the first lines
            # of the csv file are not data
            if len(data) == 63:
                # Check if the line represents column names (i.e., header)
                # If so, extract column names
                if data[0] == '"Country Name"':
                    col_names = []
                    # Get rid of quote marks in the results
                    # to make the data easier to work with
                    for i, datum in enumerate(data):
                        col_names.append(datum.replace('"',''))
                else:
                    # Transform and load the line of indicator data
                    results = transform_indicator_data(data, col_names)
                    load_indicator_data(results)

if __name__ == "__main__":

    # Run the complete ETL pipeline
    run_etl()
    print("ETL executed!")
    
    # Connect to the database
    conn = sqlite3.connect(DB_FILENAME)

    # Read the table to a dataframe
    df = pd.read_sql(f"SELECT * FROM {DB_TABLE_NAME}", con=conn)

    # Check: print some values
    print("Checking the contents of the database...")
    print(f"\nShape: \n{df.shape}")
    print(f"\nColumns: \n{df.columns}")
    print(f"\nHead (2 rows): \n{df.head(2)}")

    # Commit and close
    conn.commit()
    conn.close()
