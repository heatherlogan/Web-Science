import tweepy
import json
import pymongo
from pymongo import MongoClient
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="Web-Science")

client=MongoClient()
db=client.tweet_db
tweet_collection = db.tweet_collection
print('connected')
geo_counter = 0
london_geo = 0
city_list = []
doc_count = tweet_collection
print('tweets:', doc_count.estimated_document_count())

def findcity(coord):
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    return city

for tweet in tweet_collection.find():
    if tweet['geo'] is not None:
        geo_counter += 1
        tweet_city = findcity(tweet['geo']['coordinates'])
        city_list.append(tweet_city)
        if tweet_city == 'London':
            london_geo += 1
print('geo-tagged tweets:', geo_counter)
print('goe-tageed london:', london_geo)

