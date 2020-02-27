import tweepy
import json
import pymongo
from pymongo import MongoClient
from geopy.geocoders import Nominatim
from collections import Counter


geolocator = Nominatim(user_agent="Web-Science", timeout=10)

client=MongoClient()
db=client.tweet_db
tweet_collection = db.tweet_collection

streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets

print('connected')


def findcity(coord):
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    return city


def geo_location():

    geo_counter = 0
    london_geo = 0
    city_list = []
    doc_count = tweet_collection
    print('tweets:', doc_count.estimated_document_count())

    for tweet in tweet_collection.find():
        if tweet['geo'] is not None:
            geo_counter += 1
            tweet_city = findcity(tweet['geo']['coordinates'])
            city_list.append(tweet_city)
            if tweet_city == 'London':
                london_geo += 1

    print('geo-tagged tweets:', geo_counter)
    print('goe-tageed london:', london_geo)


def rest_stream_overlap():

    for tweet1 in streaming_tweets.find():
        for tweet2 in rest_tweets.find():
            if tweet1['_id'] == tweet2['_id']:
                print(tweet1)


def count_redundant():

    #


    pass


def count_rts_quotes():
    rts_quotes = Counter()

    for tweet in tweet_collection.find():
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

    print(rts_quotes)


def count_content_types():

    # content types = ‘photo’, ‘video’ or ‘animated_gif’
    content_types = Counter()

    for tweet in tweet_collection.find():
        content_type = ''
        print(tweet)
        try:
            entities = (tweet['extended_entities'])
            media = entities['media']
            for m in media:
                content_type = m['type']

        except:
            content_type = 'tweet only'

        content_types[content_type] += 1

    print(content_types)


if __name__=='__main__':

    rest_stream_overlap()
