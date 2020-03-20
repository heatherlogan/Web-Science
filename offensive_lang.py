import itertools
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import Counter
import pandas as pd
import re
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


client=MongoClient()
db=client.tweet_db

tweet_collection = db.tweet_collection
streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets

models = {'LR': 'Logistic regression', 'MNB': 'Multinomial Naive Bayes', 'RF': 'Random Forest'}
abbreviations = {"NOT":'Non-Offensive Tweet', 'OFF':'Offensive Tweet',
                 'TIN':'Targeted Insult', 'UNT': 'Untargeted Insult',
                 'IND':'Individual Target', 'GRP':'Group Target', 'OTH':'Other'}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, task="A", model='MNB'):

    # plots confusion matrix - taken frrom SKLEARN

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = 'Subtask {}'.format(task)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    filename='task_{}_model_{}.png'.format(task, model)
    plt.savefig('data/task3/figures/{}'.format(filename))

    plt.clf()


def plot_bar_chart(dict, task):

    # for plotting the bar chart of classified tweets

    fig, ax = plt.subplots()
    barchart = ax.bar(range(len(dict)), list(dict.values()), align='center')

    values = [abbreviations[term] for term in list(dict.keys())]

    plt.xticks(range(len(dict)), values)

    for rect in barchart:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')

    figname = "Task_{}_Results_{}".format(task,MODEL)
    title = "Task {} Results - {} Model".format(task, models.get(MODEL))
    plt.title(title)
    plt.savefig('data/task3/charts/'+figname)
    plt.clf()


def clean_tweet(tweet):

    # preprocesses: stemms, lemmatizes, tokenizes, lowercase, remove nonalphanumerric characters
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

    text = str(text).strip()
    return text.split()


def preprocess_tweet_collection():

    # for preprocessing out database of collected tweets

    csvfile = open('data/preprocessed_tweets.csv', 'w')
    csvfile.write('tweet_id,text\n')

    preprocesed_tweets = []
    sample_tweets = list(rest_tweets.find())
    sample_tweets2 = list(streaming_tweets.find())

    joined_tweets = sample_tweets + sample_tweets2

    for tweet in joined_tweets:

        if tweet['truncated']:
            try:
                text = tweet['extended_tweet']['full_text']
                text = text.replace('\n', '')

            except:
                text = tweet['text']
                text = text.replace('\n', '')
        else:
            text = tweet['text']

        preprocessed_text = ' '.join(clean_tweet(text))
        string = (str(tweet['id']) + ", " + preprocessed_text + "\n")
        csvfile.write(string)



def vectorise(train_tweets, collected_tweets):

    # TFIDF vectoriser

    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in train_tweets]
    vectorizer = vectorizer.fit(untokenized_data)
    train_vector = vectorizer.transform(untokenized_data).toarray()

    collected_tweets = [' '.join(tweet) for tweet in collected_tweets]
    collected_tweet_vector = vectorizer.transform(collected_tweets).toarray()

    return train_vector, collected_tweet_vector


def classify(train_vectors, labels, collected_tweet_vector, evaluation=True, task='A', model='MNB'):

    if evaluation==True:

        # evaluation mode: train then run on test set.
        # outputs evaluation metrics

        train_vectors, test_vectors, train_labels, test_labels = train_test_split(train_vectors, labels,
                                                                                  test_size=0.2)
        print()
        print('Training classifier .. ')

        # which classifier to use
        if model=='MNB':
            classifier = MultinomialNB(alpha=0.3)
        elif model=='LR':
            classifier = LogisticRegression(C=3, penalty='l2', multi_class='auto', solver='newton-cg')
        else:
            classifier = RandomForestClassifier(n_estimators=50, max_depth=800, min_samples_split=5)

        classifier.fit(train_vectors, train_labels)

        test_predictions = classifier.predict(test_vectors)
        accuracy = accuracy_score(test_labels, test_predictions)
        print("Test Accuracy: {:.3f}".format(accuracy))
        print("Confusion Matrix:")
        cm = (confusion_matrix(test_labels, test_predictions))
        print(cm)
        print('Report', (classification_report(test_labels, test_predictions)))

        if task=='A':
            title = 'Subtask A Results'
            classes=['Offensive', 'Non-Offensive']
        elif task=='B':
            title = 'Subtask B Results'
            classes = ['Targeted', 'Untargeted']
        else:
            title = 'Subtask B Results'
            classes = ['Individual Target', 'Group Target', 'Other']

        plot_confusion_matrix(cm, classes=classes,normalize=True,
                              title=title, task=task, model=model)
        # results = (Counter(pred))
        # print(results)
        return test_predictions

    else:

        # training classifier on training set and then on own dataset

        print('Training classifier .. ')
        # which classifier to use
        if model=='MNB':
            classifier = MultinomialNB(alpha=0.3)
        elif model=='LR':
            classifier = LogisticRegression(C=3, penalty='l2', multi_class='auto', solver='newton-cg')
        else:
            classifier = RandomForestClassifier(n_estimators=50, max_depth=800, min_samples_split=5)

        classifier.fit(train_vectors, labels)
        # use on our own tweets
        print('Vectorising collected tweets ...')
        test_vectors = collected_tweet_vector

        print('Predicting ...')
        test_predictions = classifier.predict(test_vectors)
        counts = (Counter(test_predictions))

        plot_bar_chart(counts, task)

        print('\n-RESULTS ----------------')
        for name, count in counts.items():
            print(abbreviations[name], ':', count)

        return test_predictions



def get_vectors(vectors, labels, keyword):

    # runs through the input vectors with labels and outputs vectors which match some label

    result = list()
    indicies = list()
    idx = 0

    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
            indicies.append(idx)
        idx += 1
    return result, indicies


def get_indexed_tweets(list_tweets, index_list):

    # helper to retrieve the original text from an index list, which
    # indicates the tweet vectors outputted by the previous classication task

    filtered = []

    for i, tweet in enumerate(list_tweets):
        if i in index_list:
            filtered.append(tweet)

    return filtered

def tokenize_join(tweet):
    tweet = str(tweet).strip().split()
    return ' '.join(tweet)



if __name__=="__main__":

    eval = sys.argv[1]
    MODEL = sys.argv[2]

    # setting values to use for evaluatin mode or text mode
    if eval.strip().lower()=='true':
        EVALUATION_MODE = True
        print('Running Evaluation mode on {} model'.format(models.get(MODEL)))

    else:
        EVALUATION_MODE = False
        print('Running Testing mode on {} model'.format(MODEL))

    train_path = 'data/training_data/train_file.tsv'
    test_path =  'data/test_data/test_file.txt'
    original_tweet_path = 'data/collected_tweets.csv'
    collected_tweet_path = 'data/preprocessed_tweets.csv'

    train_data = pd.read_csv(train_path, sep='\t', header=0)

    # load all necessary files

    og_data = pd.read_csv(original_tweet_path, sep=',', header=0)
    og_data['tweet_id'] = og_data['tweet_id'].astype(str)

    collected_data = pd.read_csv(collected_tweet_path, sep=',', header=0)
    collected_data['tweet_id'] = collected_data['tweet_id'].astype(str)

    collected_tweets = [tokenize_only(tweet) for tweet in collected_data['text']]

    train_tweets = train_data[["tweet"]]
    subtask_a_labels = train_data[["subtask_a"]]
    subtask_b_labels = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
    subtask_c_labels = train_data.query("subtask_b == 'TIN'")[["subtask_c"]]

    train_vector = [clean_tweet(tweet) for tweet in train_tweets['tweet'].tolist()]

    train_vector, collected_tweet_vector = vectorise(train_vector, collected_tweets)  # Numerical Vectors A
    labels_a = subtask_a_labels['subtask_a'].values.tolist() # Subtask A Labels

    print('======== Subtask A - Detecting Offensive Language ==========')
    task_a_results = classify(train_vector, labels_a, collected_tweet_vector,
                              evaluation=EVALUATION_MODE, task='A', model=MODEL)

    if EVALUATION_MODE==False:
        # reformattting dataframe to be saved as [tweet_id, original_tweet, task_a_result]
        task_a_dataframe = collected_data
        task_a_dataframe['Task A'] = task_a_results
        result_df_A = pd.merge(og_data, task_a_dataframe, on='tweet_id')
        result_df_A = result_df_A.drop(['text'], axis=1)
        result_df_A = result_df_A.set_index('tweet_id')
        result_df_A = result_df_A.drop_duplicates()
        result_df_A.to_csv('data/task3/tweets/subtask_A_tweets.csv')

    print('\n\n')

    print('======== Subtask B - Detecting Targeted Offense ==========')

    vectors_b, _ = get_vectors(train_vector, labels_a, "OFF") #
    labels_b = subtask_b_labels['subtask_b'].values.tolist()

    collected_tweet_taskb, task_b_indicies = get_vectors(collected_tweet_vector, task_a_results, "OFF") #

    train_vector_b = np.array(vectors_b)
    task_b_results = classify(train_vector_b[:], labels_b, collected_tweet_taskb,
                              evaluation=EVALUATION_MODE, task='B', model=MODEL)

    if EVALUATION_MODE==False:
        offensive_tweets = get_indexed_tweets(collected_tweets, task_b_indicies)
        offensive_tweets = [' '.join(tweet) for tweet in offensive_tweets]

        # saving task B to CSV

        task_b_df = pd.DataFrame()
        task_b_df['text'] = offensive_tweets
        task_b_df['Task B'] = task_b_results
        task_b_df = task_b_df.drop_duplicates('text')
        task_b_df.index.name = 'idx'
        task_b_df.to_csv('data/task3/tweets/subtask_B_tweets.csv')

    print('\n\n')
    print('======== Subtask C - Identifying Target of Offenses  ==========')

    train_vector_c, _ = get_vectors(train_vector_b, labels_b, "TIN")  # Numerical Vectors C
    train_vector_c = np.array(train_vector_c)

    labels_c = subtask_c_labels['subtask_c'].values.tolist()  # Subtask A Labels

    collected_tweet_taskc, c_indicies = get_vectors(collected_tweet_taskb, task_b_results, "TIN")  #

    task_c_results = classify(train_vector_c[:], labels_c, collected_tweet_taskc,
                              evaluation=EVALUATION_MODE, task='C', model=MODEL)

    if EVALUATION_MODE==False:
        # g
        targeted_insults = get_indexed_tweets(offensive_tweets, c_indicies)
        targeted_insults = [''.join(tweet) for tweet in targeted_insults]

        # saving task C to csv
        task_c_df = pd.DataFrame()
        task_c_df['text'] = targeted_insults
        task_c_df['Task C'] = task_c_results
        task_c_df = task_c_df.drop_duplicates('text')
        task_c_df.index.name = 'idx'
        task_c_df.to_csv('data/task3/tweets/subtask_C_tweets.csv')

        # ROUGH FIX FOR TEMPORARY FILES

        pp_tweets = pd.read_csv('data/preprocessed_tweets.csv')
        a_results = pd.read_csv('data/task3/tweets/subtask_A_tweets.csv')
        b_results = pd.read_csv('data/task3/tweets/subtask_B_tweets.csv')
        b_results = b_results.set_index('idx')
        b_results = b_results.drop_duplicates('text')
        c_results = pd.read_csv('data/task3/tweets/subtask_C_tweets.csv')
        c_results = c_results.set_index('idx')

        pp_tweets['text'] = pp_tweets['text'].apply(tokenize_join)
        b_results['text'] = b_results['text'].apply(tokenize_join)
        task_c_df['text'] = task_c_df['text'].apply(tokenize_join)

        merged1 = pd.merge(pp_tweets, b_results, on='text')
        merged1 = merged1.drop_duplicates('text')

        merged2 = pd.merge(merged1, c_results, on='text')
        merged2 = merged2.drop_duplicates('text')

        merged1 = merged1.drop('text', axis=1)
        merged2 = merged2.drop('text', axis=1)

        full_tweets_B = pd.merge(a_results, merged1, on='tweet_id')
        full_tweets_B = full_tweets_B.drop_duplicates('tweet_id')
        full_tweets_B.to_csv('data/task3/tweets/subtask_B_tweets.csv')

        full_tweets_C = pd.merge(a_results, merged2, on='tweet_id')
        full_tweets_C = full_tweets_C.drop_duplicates('tweet_id')
        full_tweets_C.to_csv('data/task3/tweets/subtask_C_tweets.csv')
