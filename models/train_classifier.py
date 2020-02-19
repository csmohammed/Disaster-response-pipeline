# import libraries
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger', 'stopwords'])
from sklearn.pipeline import Pipeline , FeatureUnion
import re
import pickle
import numpy as np
import sys
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import operator
from pprint import pprint
from nltk.corpus import stopwords


def load_data(database_filepath):
    """
    this function is used to load the data from the database    
    
    Input:
    database file path 
    
    Output:
    x: which contains messages column  
    Y: which contains all values of 36 features
    category_names: which contains name of 36 features 
    
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    Y = df.loc[:,'related':'direct_report']
    category_names = Y.columns   
    # Y['related'] contains three distinct values
    # mapping extra values to `1`
    Y['related']=Y.related.replace(2, 1)
    return X, Y, category_names 

def tokenize(text):
    """
    this function is used to clean data     
    
    Input:
    text messages 
    
    Output:
    text: which contains list and clean text 
    
    """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words

def build_model():
    """
    this function is used to build pipeline model      
    
    Input:
    NONE 
    
    Output:
    model: after apply  grid search
    
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier())])
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    this function is used toevaluate the model on the testing      
    
    Input:
      model:          the trained model 
      X_test:         the text to be tested 
      Y_test:         actual features to compare against 
      category_names: features
    Output:
    NONE
    
    """
    y_pred = model.predict(X_test)

    # print classification & accuracy score
    print(classification_report(np.hstack(Y_test.values), np.hstack(y_pred), target_names=category_names))
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_file_path):
    """
    this function is used toevaluate the model on the testing      
    
    Input:
      model:           the trained model 
      model_file_path: the path to save the PKL trained model
    Output:
    NONE
    
    """
    pickle.dump(model, open(model_file_path, 'wb'))


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