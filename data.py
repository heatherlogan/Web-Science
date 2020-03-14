import numpy as np
import tweepy
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



geolocator = Nominatim(user_agent="Web-Science", timeout=None)

client=MongoClient()
db=client.tweet_db

tweet_collection = db.tweet_collection
streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets

sample_tweets = list(rest_tweets.find())
sample_tweets2 = list(streaming_tweets.find())

joined_tweets = sample_tweets + sample_tweets2


def findcity(coord):
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    return city


def geo_location():
    tweet_df = pd.DataFrame(joined_tweets)
    geo_counter = 0
    london_geo = 0
    tweet_df = tweet_df.dropna(subset=['geo'])
    tweet_df['locname'] = tweet_df.place.apply(lambda x: x.get('name'))
    tweet_df = tweet_df[tweet_df['locname'].str.match('London') == True]

    li2 = tweet_df['_id'].tolist()
    print(li2)
    li = []
    for i in range(0, len(li)):
        #fix here
        s = li[i]['_id'].getTimestamp()
        li.append('%s:%s' % (s.hour, s.minute))
    conv_time = [datetime.strptime(i, "%H:%M") for i in li]
    df = pd.DataFrame(conv_time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])
    times = [t.hour + t.minute / 60. for t in df['time']]
    print(times)
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


    print('Geo-tagged tweets:', geo_counter)
    print('Gee-tagged london:', london_geo)


def rest_stream_overlap():
    stream_ids = [tweet['id'] for tweet in streaming_tweets.find()]
    rest_ids = [tweet['id'] for tweet in rest_tweets.find()]

    overlap = ((list(set(stream_ids) & set(rest_ids))))

    redundant_stream = [i for i,c in Counter(stream_ids).items() if c>1]
    redundant_rest = [i for i,c in Counter(rest_ids).items() if c>1]

    print('Redundant Stream Tweets', len(redundant_stream))
    print('Redundant REST Tweets', len(redundant_rest))
    print('Number of overlapping Stream/REST Tweets: ', len(overlap))




def count_rts_quotes():
    rts_quotes = Counter()

    for tweet in joined_tweets:
        try:
            retweeted = tweet['retweeted_status']
            rts_quotes['rt'] += 1
        except:
            pass
        try:
            quoted = tweet['quoted_status']
            rts_quotes['quote'] += 1
        except:
            pass

    print('RT\'d tweets: ', rts_quotes.get('rt'))
    print('Quoted tweets: ', rts_quotes.get('quote'))


def count_content_types():
    # content types = ‘photo’, ‘video’ or ‘animated_gif’
    content_types = Counter()


    for tweet in joined_tweets:
        content_type = ''
        try:
            entities = (tweet['extended_entities'])
            media = entities['media']
            for m in media:
                content_type = m['type']

        except:
            content_type = 'tweet only'

        content_types[content_type] += 1

    print('Content Types: ', [ '{}: {}'.format(x,y) for x,y in content_types.items()])


def graph(data):
    #data.reverse()
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



if __name__=='__main__':
    print('Total Tweets Collected:', len(joined_tweets))
    print('Tweets collected using Stream :', streaming_tweets.estimated_document_count())
    print('Tweets collected using REST API :', rest_tweets.estimated_document_count())

    geo_location()
   # rest_stream_overlap()
   # count_rts_quotes()
   # count_content_types()

