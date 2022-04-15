# import libraries
import sys

import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

## nlp specific libraries
import nltk
import re
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

## ml pipeline specific libraries

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Loads dataset from the given database_filepath.
       Return X with the input message, y that contains the features, and the category/feature name
    """

    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql('select * from messages', engine)
    
    ## load X, and y
    
    all_cols = df.columns.tolist()
    rel_cols = all_cols[4:]

    X = df['message']
    y = df[rel_cols]
    
#     y['related'].replace(2, 1, inplace=True)
    category_names = y.columns.tolist()
    return X, y, category_names    


def tokenize(text):
    """Given an input text, cleans it, removes stop words, and splits into words."""
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens]
    return cleaned_tokens


def build_model():
    """Defines a pipeline and uses grid search to find optimum parameters."""
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Computes the precision, recall, and f1-score for each of the category."""
    
    ## make predictions and converts it to a dataframe
    y_pred = model.predict(X_test)
    df_y_pred = pd.DataFrame(y_pred, columns=category_names)
    
    evaluation = {}
    
    for column in Y_test.columns:
        evaluation[column] = []
        evaluation[column].append(precision_score(Y_test[column], df_y_pred[column], average="weighted"))
        evaluation[column].append(recall_score(Y_test[column], df_y_pred[column], average="weighted"))
        evaluation[column].append(f1_score(Y_test[column], df_y_pred[column], average="weighted"))
        
    print(pd.DataFrame(evaluation))    


def save_model(model, model_filepath):
    """Saves the passed trained model to specific filepath."""

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()