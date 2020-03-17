import numpy as np
import tweep
import json
import pymongo
import re

from pandas import DataFrame
from pandas.io.json import json_normalize
from pymongo import MongoClient
from geopy.geocoders import Nominatim
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime
from matplotlib import dates as mdates


client = MongoClient()
db = client.tweet_db1

tweet_collection = db.tweet_collection
streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets

sample_tweets = list(rest_tweets.find())
sample_tweets2 = list(streaming_tweets.find())

joined_tweets = sample_tweets + sample_tweets2


def geo_location(data):
    li = []
    for i in range(0, len(data)):
        s = data[i]['_id'].generation_time
        li.append(s)
    tweet_df = pd.DataFrame(data)
    tweet_df['time'] = li
    tweet_df = tweet_df.dropna(subset=['geo'])
    tweet_df['locname'] = tweet_df.place.apply(lambda x: x.get('name'))

    tweet_df = tweet_df[tweet_df['locname'].str.match('London') == True]
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
    plt.title('London geo tagged tweets in 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def rest_stream_overlap(data):
    li = []
    for i in range(0, len(data)):
        s = data[i]['_id'].generation_time
        li.append(s)
    tweet_df = pd.DataFrame(data)
    tweet_df['time'] = li
    tweet_df = tweet_df[tweet_df.duplicated(['id'], keep=False)]

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


def count_rts_quotes(data):
    li = []
    for i in range(0, len(data)):
        s = data[i]['_id'].generation_time
        li.append(s)
    tweet_df = pd.DataFrame(data)
    tweet_df['time'] = li

    #tweet_df.dropna(subset=['retweeted_status', 'quoted_status'], thresh=1, inplace=True)
    #tweet_df.dropna(subset=['retweeted_status'], inplace=True)
    tweet_df.dropna(subset=['quoted_status'], inplace=True)

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
    plt.title('Amount of quotes 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def count_content_types(data):
    # content types = ‘photo’, ‘video’ or ‘animated_gif’
    li = []
    for i in range(0, len(data)):
        s = data[i]['_id'].generation_time
        li.append(s)
    tweet_df = pd.DataFrame(data)
    tweet_df['time'] = li
    tweet_df.dropna(subset=['extended_entities'], inplace=True)
    tweet_df['media'] = tweet_df.extended_entities.apply(lambda x: x.get('media')[0].get('type'))
    print(tweet_df.groupby(['media']).count())

    #animated_gif, photo, video
    tweet_df = tweet_df[tweet_df['media'].str.match('video') == True]


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
    plt.title('Amount of video media 10 minute bins')
    newlabels = []
    for edge in bins:
        h, m = divmod(edge % 12, 1)
        newlabels.append('{0:01d}:{1:02d}'.format(int(h), int(m * 60)))

    ax.set_xticklabels(newlabels)
    plt.show()


def graph(data):
    # data.reverse()
    li = []
    for i in range(0, len(data)):
        s = data[i]['_id'].generation_time
        li.append('%s:%s' % (s.hour, s.minute))

    conv_time = [datetime.strptime(i, "%H:%M") for i in li]
    df = pd.DataFrame(conv_time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])
    times = [t.hour + t.minute / 60. for t in df['time']]
    tinterval = 10.
    lowbin = np.min(times) - np.fmod(np.min(times) - np.floor(np.min(times)), tinterval / 60.)
    highbin = np.max(times) - np.fmod(np.max(times) - np.ceil(np.max(times)), tinterval / 60.)
    bins = np.arange(lowbin, highbin, tinterval / 60.)
    bins = bins[:-1]
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
    print('Total Tweets Collected:', len(joined_tweets))
    print('Tweets collected using Stream :', streaming_tweets.estimated_document_count())
    print('Tweets collected using REST API :', rest_tweets.estimated_document_count())

    geo_location(joined_tweets)

    # rest_stream_overlap(joined_tweets)
    #count_rts_quotes(joined_tweets)
    #count_content_types(joined_tweets)
