from collections import Counter
import pandas as pd
import re
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import copy
import numpy as np


client=MongoClient()
db=client.tweet_db

tweet_collection = db.tweet_collection
streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets

abbreviations = {"NOT":'Non-Offensive Tweet', 'OFF':'Offensive Tweet', 'TIN':'Targeted Insult',
                 'UNT': 'Untargeted Insult', 'IND':'Individual Target', 'GRP':'Group Target', 'OTH':'Other'}

def clean_tweet(tweet):
    pp_tokens = []
    text = str(tweet).split()
    for token in text:
        if not token.startswith('@') and token != 'RT' and token not in stopwords and not token.startswith('http'):
            token = token.lower()
            token = re.sub(r'\W+', '', token)
            token = wordnet_lemmatizer.lemmatize(token)
            token = lancaster_stemmer.stem(token)
            pp_tokens.append(token)

    return pp_tokens

def tokenize_only(text):

    text = text.strip()
    return text.split()



def preprocess_tweet_collection():

    # for preprocessing out database of collected tweets

    csvfile = open('preprocessed_tweets.csv', 'w')

    preprocesed_tweets = []
    sample_tweets = list(rest_tweets.find())
    sample_tweets2 = list(streaming_tweets.find())

    joined_tweets = sample_tweets + sample_tweets2

    for tweet in joined_tweets:

        if tweet['truncated']:
            try:
                text = tweet['extended_tweet']['full_text']
            except:
                text = tweet['text']
        else:
            text = tweet['text']

        preprocessed_text = clean_tweet(text)
        string = (str(tweet['id']) + ", " + preprocessed_text + "\n")
        csvfile.write(string)



def vectorise(train_tweets, collected_tweets):

    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in train_tweets]
    vectorizer = vectorizer.fit(untokenized_data)
    train_vector = vectorizer.transform(untokenized_data).toarray()

    collected_tweets = [' '.join(tweet) for tweet in collected_tweets]
    collected_tweet_vector = vectorizer.transform(collected_tweets).toarray()

    return train_vector, collected_tweet_vector


def classify(train_vectors, labels, collected_tweet_vector, evaluation=True):

    if evaluation==True:

        train_vectors, test_vectors, train_labels, test_labels = train_test_split(train_vectors, labels,
                                                                                  test_size=0.2)
        print()
        print('Training classifier .. ')
        classifier = MultinomialNB(alpha=0.3)

        classifier.fit(train_vectors, train_labels)
        # pred = classifier.predict(train_vectors)
        #
        # accuracy = accuracy_score(labels, pred)
        # print("Training Accuracy: {:.3f}".format(accuracy))
        test_predictions = classifier.predict(test_vectors)
        accuracy = accuracy_score(test_labels, test_predictions)
        print("Test Accuracy: {:.3f}".format(accuracy))
        print("Confusion Matrix:")
        print(confusion_matrix(test_labels, test_predictions))
        # results = (Counter(pred))
        # print(results)
    else:

        print('Training classifier .. ')
        classifier = MultinomialNB(alpha=0.3)
        classifier.fit(train_vectors, labels)
        # use on our own tweets
        print('Vectorising collected tweets ...')
        test_vectors = collected_tweet_vector

        print('Predicting ...')
        test_predictions = classifier.predict(test_vectors)
        counts = (Counter(test_predictions))

        print('\n ----------- RESULTS -----------')
        for name, count in counts.items():
            print('\t', abbreviations[name], ':', count)

        return test_predictions



def get_vectors(vectors, labels, keyword):
    result = list()
    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
    return result



if __name__=="__main__":

    train_path = 'data/training_data/train_file.tsv'
    test_path =  'data/test_data/test_file.txt'
    collected_tweet_path = 'data/preprocessed_tweets.csv'

    train_data = pd.read_csv(train_path, sep='\t', header=0)
    collected_data = pd.read_csv(collected_tweet_path, sep=',', header=0)

    collected_tweets = [tokenize_only(tweet) for tweet in collected_data['tweet']]

    tweets = train_data[["tweet"]]
    subtask_a_labels = train_data[["subtask_a"]]
    subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
    subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]

    vector = [clean_tweet(tweet) for tweet in tweets['tweet'].tolist()]

    print('Vectorising data ... \n')
    train_vector, collected_tweet_vector = vectorise(vector, collected_tweets)  # Numerical Vectors A
    labels_a = subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

    print('======== Subtask A - Detecting Offensive Language ==========')
    task_a_results = classify(train_vector, labels_a, collected_tweet_vector, evaluation=False)

    print('\n\n')


    print('======== Subtask B - Detecting Targeted Offense ==========')

    vectors_b = get_vectors(train_vector, labels_a, "OFF") #
    labels_b = subtask_b_labels['subtask_b'].values.tolist()
    collected_tweet_taskb = get_vectors(collected_tweet_vector, task_a_results, "OFF") #

    train_vector_b = np.array(vectors_b)
    task_b_results = classify(train_vector_b[:], labels_b, collected_tweet_taskb,  evaluation=False)

    print('\n\n')
    print('======== Subtask C - Identifying Target of Offenses  ==========')

    train_vector_c = get_vectors(train_vector_b, labels_b, "TIN") # Numerical Vectors C
    train_vector_c = np.array(train_vector_c)
    labels_c = subtask_c_labels['subtask_c'].values.tolist() # Subtask A Labels

    collected_tweet_taskc = get_vectors(collected_tweet_taskb, task_b_results, "TIN") #

    task_c_results = classify(train_vector_c[:], labels_c, collected_tweet_taskc, evaluation=False)


    # # writing to results files
    # raw_tweets = pd.read_csv('data/collected_tweets.csv', sep=',', header=0)
    # raw_tweets['Task A'] = pd.Series(task_a_results)
    # raw_tweets.to_csv('data/results/task_a_results.csv')