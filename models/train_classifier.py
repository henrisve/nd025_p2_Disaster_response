import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import re
import pickle
from nltk.corpus import stopwords
import os
import errno

def load_data(database_filepath):
# load data from database
    # Make sure the file actually exist, otherwise an 
    # empty will be created
    if not os.path.exists(database_filepath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), database_filepath)
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = "table1"
    sql_query = f'Select * From {table_name}'
    df = pd.read_sql(sql_query,engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = list(Y)
    print(category_names)
    print(X.shape, Y.shape)
    return X, Y, category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    text = re.sub(url_regex, 'urlplaceholder',text)
    
    #clean up and normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = nltk.tokenize.word_tokenize(text)

    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # initiate lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer() 

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
    ])
    parameters = {
        'clf__estimator__n_estimators': [1, 2],
        #'clf__estimator__min_samples_split': [2, 4],
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i, pred in enumerate(np.swapaxes(y_pred,0,1)):
        print(classification_report(Y_test.iloc[:,i].values, pred))
        print("----------------------------------")


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    print("start")
    main()