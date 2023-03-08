"""This module shows an exemplary Machine Learning pipeline
in which NLP processing is done to classify corporate messages
according to their content.

The following methods/techniques are used:

- Encoding detection
- Custom Scikit-Learn transformers
- NLP pipeline: cleaning, normalization, tokenization, lemmatization
- Scikit-Learn Pipeline
- FetureUnion
- GridSearchCV

Using Pipeline along with GridSearchCV has several advantages:

- More compact code.
- Repetitive steps automated.
- Code easier to understand and modify.
- We can apply `GridSearchCV` to the complete `Pipeline`,
    so we optimize transformer parameters, if necessary.
- We prevent data leakage, because the transformers in
    GridSearchCV are fit in each fold with a different
    subset, so the complete training data is not leaked.

To use this file, check that DATA_FILENAME points
to the correct file path and run the script:

    $ python ml_nlp_pipeline.py

Note that you need to run the script in an environment
where all necessary packages have been installed:

    numpy, pandas, nltk, sklearn, chardet

Source: Udacity Data Science Nanodegree case study,
link to original file:

    https://github.com/mxagar/data_science_udacity/blob/main/03_DataEngineering/lab/ML_Pipelines/ml_nlp_pipeline.py

Author: Mikel Sagardia
Date: 2023-03-08
"""
import re
import numpy as np
import pandas as pd
import chardet

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Constants
DATA_FILENAME = "./data/corporate_messaging.csv"
URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer. Guidelines:

    - Always define fit() and transform()
    and return self if there is nothing to return. 
    - Define __init__() if attributes need to be stored.
    
    This custom transformer returns True if any of the sentences
    in the text/message starts with a verb; otherwise, False."""
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def detect_encoding():
    """Detect encoding of the dataset (default: 'utf_8').
    
    Args: None
    
    Returns:
        encoding (str): encoding as string
    """
    # !pip install chardet
    encoding = 'utf_8'
    with open(DATA_FILENAME, 'rb') as file:
        encoding = chardet.detect(file.read())['encoding']

    return encoding


def load_data():
    """Load dataset.
    
   The dataframe has 11, but only 3 columns are used:
   
   - `text`: message to be classfied.
   - `category`: target message category:
        `Information`, `Action`, `Dialogue` and `Exclude`;
        (the last is not taken).
   - `category:confidence`: the confidence of the category;
        only rows with confidence 1 are taken.  
    
    Args: None
    
    Returns:
        X, y (tuple): X is a 1D array of messages,
            y is a 1D array of text categories."""
    encoding = detect_encoding()
    df = pd.read_csv(DATA_FILENAME, encoding=encoding) # 'mac_roman'
    df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
    X = df.text.values
    y = df.category.values

    return X, y


def tokenize(text):
    """Perform the NLP:
    
    - Clean
    - Normalize
    - Tokenize
    - Lemmatize
    
    Args:
        text (string): message
    
    Returns:
        clean_tokens (list): list of processed lemmas
    """
    detected_urls = re.findall(URL_REGEX, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """The Pipeline is defined, as well as
    the GridSearCV object.
    
    Args: None
    
    Returns:
        grid (object): GridSearchCV object
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }

    grid = GridSearchCV(pipeline, param_grid=parameters)

    return grid


def display_results(grid, y_test, y_pred):
    """Display evaluation results.
    
    Args:
        grid (object): trained Pipeline embedded in a GridSearchCV
        y_test (np.array): real target values
        y_pred (np.array): predicted target values
    
    Returns: None
    """
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", grid.best_params_)


def train_pipeline():
    """Entire training pipeline is executed:
    
    - load_dataset()
    - build_model() 
    - Train (fit) pipeline/model
    - Evaluate and display_results()
    
    Args: None
    
    Returns:
        model (object): trained Pipeline embedded in a GridSearchCV
    """
    print("\nLoading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("\nTraining model...")
    model = build_model()
    model.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    display_results(model, y_test, y_pred)

    return model

if __name__ == "__main__":
    """Entire training pipeline is executed
    by calling to train_pipeline()."""
    _ = train_pipeline()
