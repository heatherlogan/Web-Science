import tweepy
import json
import pymongo
from pymongo import MongoClient


CONSUMER_KEY = "TYPVBk4kO17UMfDZSOURAGL4E"
CONSUMER_SECRET = "0Kq5zHsOb4lNfDGGbwOmhVaXTkpwS1DZiA5kBgpFt6L3OilJdt"
ACCESS_TOKEN = "431764921-CaJPtXrUEu6PoYsyYozOFMrAu7rbSBZKUOppVpBq"
ACCESS_TOKEN_SECRET = "N1qZUsVN5h4vz31MdIAwJWIkO6oJCJUBx6PSlIXRvKqSj"


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

client=MongoClient()
db=client.tweet_db
tweet_collection = db.tweet_collection
print('connected')


data = api.rate_limit_status()
print(data)
max_tweets = 10

def RESTtimeline(name):
    timeline = tweepy.Cursor(api.user_timeline,screen_name=name, lang='en', include_rts=False).items(1000)
    for item in timeline:
        all_data=item._json
        tweet_collection.insert_one(all_data)


def user_restprobe():
    users = ['jeremycorbyn', 'BorisJohnson', 'NicolaSturgeon']
    for name in users:
        RESTtimeline(name)


def keyword_restprobe():
    query = ['SNP', 'Labour', 'Tories']
    for tweet in tweepy.Cursor(api.search, q=query, lang='en', include_entities=True).items():
        tweet_collection.insert_one(tweet)









