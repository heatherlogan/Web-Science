import pandas as pd
import re, nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from data import *
stopwords = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_tweets():

    csvfile = open('preprocessed_tweets.csv', 'w')

    preprocesed_tweets = []
    sample_tweets = list(rest_tweets.find())
    sample_tweets2 = list(streaming_tweets.find())

    joined_tweets = sample_tweets + sample_tweets2

    for tweet in joined_tweets:
        preprocessed_text = ""
        text = (tweet['text'].split(' '))
        for token in text:
            if not token.startswith('@') and token != 'RT' and token not in stopwords and not token.startswith('http'):
                token = token.lower()
                token = re.sub(r'\W+', '', token)
                token = wordnet_lemmatizer.lemmatize(token)
                token = lancaster_stemmer.stem(token)
                preprocessed_text += " " + (token)

        string = (str(tweet['id']) + ", " + preprocessed_text + "\n")
        csvfile.write(string)


def train_classifier():


    pass





def detect_language():

    # use a dataset to classify the tweet into offensive/not offensive



    pass



def classify_offense_type():


    pass



def determine_target():


    pass


if __name__=="__main__":
    preprocess_tweets()