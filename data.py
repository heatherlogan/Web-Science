import numpy as np
from jsoncomment import JsonComment

from pymongo import MongoClient
import json
import pymongo
from json import JSONEncoder
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime
from matplotlib import dates as mdates



def geo_location(tweet_df):
    tweet_df = tweet_df.dropna(subset=['geo'])
    tweet_df = tweet_df.dropna(subset=['place'])
    tweet_df['locname'] = tweet_df.place.apply(lambda x: x.get('name'))

    tweet_df['locname'] = tweet_df.place.apply(lambda x: x.get('name'))
    tweet_df = tweet_df[tweet_df['locname'].str.match('London') == True]
    tweet_df['time'] = pd.to_datetime(tweet_df['time'])
    times = [t.hour + t.minute / 60. for t in tweet_df['time']]
    print('geo', len(tweet_df))
    tinterval = 10.
    lowbin = np.min(times) - np.fmod(np.min(times) - np.floor(np.min(times)), tinterval / 60.)
    highbin = np.max(times) - np.fmod(np.max(times) - np.ceil(np.max(times)), tinterval / 60.)
    bins = np.arange(lowbin, highbin, tinterval / 60.)
    plt.hist(times, bins=bins, edgecolor='black')
    ax = plt.gca()
    ax.set_xticks(bins)
    plt.xlabel('time in PM')
    plt.ylabel('amount of tweets')
    plt.title('London geo tagged tweets in 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def rest_stream_overlap(tweet_df):
    tweet_df = tweet_df[tweet_df.duplicated(['id'], keep=False)]
    print('over', len(tweet_df))

    tweet_df['time'] = pd.to_datetime(tweet_df['time'])
    times = [t.hour + t.minute / 60. for t in tweet_df['time']]
    tinterval = 10.
    lowbin = np.min(times) - np.fmod(np.min(times) - np.floor(np.min(times)), tinterval / 60.)
    highbin = np.max(times) - np.fmod(np.max(times) - np.ceil(np.max(times)), tinterval / 60.)
    bins = np.arange(lowbin, highbin, tinterval / 60.)
    plt.hist(times, bins=bins, edgecolor='black')
    ax = plt.gca()
    ax.set_xticks(bins)
    plt.xlabel('time in PM')
    plt.ylabel('amount of tweets')
    plt.title('Duplicate tweets 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def count_rts_quotes(tweet_df, counter):
    if counter == 0:
        tweet_df.dropna(subset=['retweeted_status', 'quoted_status'], thresh=1, inplace=True)
    if counter == 1:
        tweet_df.dropna(subset=['retweeted_status'], inplace=True)
    if counter == 2:
        tweet_df.dropna(subset=['quoted_status'], inplace=True)
    print('rt', len(tweet_df), counter)
    tweet_df['time'] = pd.to_datetime(tweet_df['time'])
    times = [t.hour + t.minute / 60. for t in tweet_df['time']]
    tinterval = 10.
    lowbin = np.min(times) - np.fmod(np.min(times) - np.floor(np.min(times)), tinterval / 60.)
    highbin = np.max(times) - np.fmod(np.max(times) - np.ceil(np.max(times)), tinterval / 60.)
    bins = np.arange(lowbin, highbin, tinterval / 60.)
    plt.hist(times, bins=bins, edgecolor='black')
    ax = plt.gca()
    ax.set_xticks(bins)
    #ax.set_ylim([0, 20000])
    plt.xlabel('time in PM')
    plt.ylabel('amount of tweets')
    if counter == 0:
        plt.title('Amount of quotes & retweets 10 minute bins')
    if counter == 1:
        plt.title('Amount of retweets 10 minute bins')
    if counter == 2:
        plt.title('Amount of quotes 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def count_content_types(tweet_df, counter):
    # content types = ‘photo’, ‘video’ or ‘animated_gif’

    tweet_df.dropna(subset=['extended_entities'], inplace=True)
    tweet_df['media'] = tweet_df.extended_entities.apply(lambda x: x.get('media')[0].get('type'))
    print(tweet_df.groupby(['media']).count())

    #animated_gif, photo, video
    if counter == 0:
        tweet_df = tweet_df[tweet_df['media'].str.match('photo') == True]
    if counter == 1:
        tweet_df = tweet_df[tweet_df['media'].str.match('video') == True]
    if counter == 2:
        tweet_df = tweet_df[tweet_df['media'].str.match('animated_gif') == True]

    print('cont', len(tweet_df), counter)
    tweet_df['time'] = pd.to_datetime(tweet_df['time'])
    times = [t.hour + t.minute / 60. for t in tweet_df['time']]
    tinterval = 10.
    lowbin = np.min(times) - np.fmod(np.min(times) - np.floor(np.min(times)), tinterval / 60.)
    highbin = np.max(times) - np.fmod(np.max(times) - np.ceil(np.max(times)), tinterval / 60.)
    bins = np.arange(lowbin, highbin, tinterval / 60.)
    plt.hist(times, bins=bins, edgecolor='black')
    ax = plt.gca()
    ax.set_xticks(bins)
    # ax.set_ylim([0, 20000])
    plt.xlabel('time in PM')
    plt.ylabel('amount of tweets')
    if counter == 0:
        plt.title('Amount of photo media 10 minute bins')
    if counter == 1:
            plt.title('Amount of video media 10 minute bins')
    if counter == 2:
        plt.title('Amount of animated gif media 10 minute bins')

    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def graph(data):
    # data.reverse()
    # li = []
    # for i in range(0, len(data)):
    #     s = data[i]['_id'].generation_time
    #     li.append('%s:%s' % (s.hour, s.minute))
    #conv_time = [datetime.strptime(i, "%H:%M") for i in li]

    #Comment this line out for DB
    conv_time = [datetime.strptime(i, "%m/%d/%Y, %H:%M:%S") for i in data]



    df = pd.DataFrame(conv_time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])
    times = [t.hour + t.minute / 60. for t in df['time']]
    tinterval = 10.
    lowbin = np.min(times) - np.fmod(np.min(times) - np.floor(np.min(times)), tinterval / 60.)
    highbin = np.max(times) - np.fmod(np.max(times) - np.ceil(np.max(times)), tinterval / 60.)
    bins = np.arange(lowbin, highbin, tinterval / 60.)
    bins = bins
    plt.hist(times, bins=bins, edgecolor='black')
    ax = plt.gca()
    ax.set_xticks(bins)
    plt.xlabel('time in PM')
    plt.ylabel('amount of tweets')
    plt.title('1 hour of tweets in 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()




if __name__ == '__main__':
    # To run from DB rather than json file, uncomment below code & uncomment code in the graph function


    # client = MongoClient()
    # db = client.tweet_db
    # streaming_tweets = db.streaming_tweets
    # rest_tweets = db.rest_tweets
    # print('Database created')
    #
    # sample_tweets = list(rest_tweets.find())
    # sample_tweets2 = list(streaming_tweets.find())
    # joined_tweets = sample_tweets + sample_tweets2
    # print('Total Tweets Collected:', len(joined_tweets))
    # print('Tweets collected using Stream :', streaming_tweets.estimated_document_count())
    # print('Tweets collected using REST API :', rest_tweets.estimated_document_count())
    # for i in range(0, len(joined_tweets)):
    #     s = joined_tweets[i]['_id'].generation_time
    #     li.append(s)
    # tweetdf = pd.DataFrame(joined_tweets)
    #tweetdf['time'] = li
    #graph(joined_tweets)

    #Comment these lines out for DB
    with open('sample_tweets.json') as f:
        joined_tweets = json.load(f)
    tweetdf = pd.DataFrame(joined_tweets)
    graph(tweetdf['time'])



    geo_location(tweetdf)
    rest_stream_overlap(tweetdf)
    for i in range(0,3):
       count_rts_quotes(tweetdf, i)

    for i in range(0, 3):
        count_content_types(tweetdf, i)

