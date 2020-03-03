import tweepy
import json
import pymongo
from pymongo import MongoClient
from geopy.geocoders import Nominatim
from collections import Counter

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

    geo_counter = 0
    london_geo = 0
    city_list = []

    for tweet in streaming_tweets.find():
        if tweet['geo'] is not None:
            geo_counter += 1
            tweet_city = findcity(tweet['geo']['coordinates'])
            city_list.append(tweet_city)
            if tweet_city == 'London':
                london_geo += 1

    for tweet in rest_tweets.find():
        if tweet['geo'] is not None:
            geo_counter += 1
            tweet_city = findcity(tweet['geo']['coordinates'])
            city_list.append(tweet_city)
            if tweet_city == 'London':
                london_geo += 1

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


if __name__=='__main__':

    print('Total Tweets Collected:', len(joined_tweets))
    print('Tweets collected using Stream :', streaming_tweets.estimated_document_count())
    print('Tweets collected using REST API :', rest_tweets.estimated_document_count())

    geo_location()
    rest_stream_overlap()
    count_rts_quotes()
    count_content_types()

