import json
import plotly
import pandas as pd
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import operator

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from pprint import pprint
from nltk.corpus import stopwords


app = Flask(__name__)

## replace this function to remove stop words
def tokenize(text):
    
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

"""
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
"""
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    ##################################      ############### mohammed
    categories_proportion = df.loc[:,'related':'direct_report'].mean().sort_values(ascending = False)        
    categories = list(df.loc[:,'related':'direct_report'].columns)                             
    all_words=[]                               
                                                           
    for text in df['message'].values:
        clean_tokenized_ = tokenize(text)
        all_words.extend(clean_tokenized_)
### to have words with number of counts       
    word_counts = Counter(all_words)      
                                                          
    #  to sort the dictionary
    sorted_word_count = dict(sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True))   
    ####                                                    
    counter_ =0
    top_10_count = {}

    for key_dic,value_dic in sorted_word_count.items():
        top_10_count[key_dic]=value_dic
        counter_+=1
        if counter_==10:
            break
    words=list(top_10_count.keys())
    pprint(words)
    count_proportion=100*np.array(list(top_10_count.values()))/df.shape[0]
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=categories_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Proportion",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=count_proportion
                )
            ],

            'layout': {
                'title': 'Frequency of top 10 words <br> as percentage',
                'yaxis': {
                    'title': 'Occurrence<br>(Out of 100)',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 10 words',
                    'automargin': True
                }
            }
        }
    ]
        
    
    
    ##################################    ############### mohammed
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()