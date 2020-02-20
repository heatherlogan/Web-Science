from __future__ import print_function
import tweepy
import json
import pymongo
from pymongo import MongoClient


# TWITTER API Authentication

CONSUMER_KEY = "TYPVBk4kO17UMfDZSOURAGL4E"
CONSUMER_SECRET = "0Kq5zHsOb4lNfDGGbwOmhVaXTkpwS1DZiA5kBgpFt6L3OilJdt"
ACCESS_TOKEN = "431764921-CaJPtXrUEu6PoYsyYozOFMrAu7rbSBZKUOppVpBq"
ACCESS_TOKEN_SECRET = "N1qZUsVN5h4vz31MdIAwJWIkO6oJCJUBx6PSlIXRvKqSj"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Database Setup

client=MongoClient()
db=client.tweet_db
tweet_collection = db.tweet_collection
print('Database created')


class StdOutListener(tweepy.StreamListener):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.limit = 10

    def on_connect(self):
        print("Connected")


    def on_error(self, status_code):
        print('Error: ' + repr(status_code))
        return False

    def on_data(self, data):
        try:
            datajson = json.loads(data)
            # print(datajson)
            print('User:', datajson['user']['screen_name'])
            print('Tweet: ', datajson['text'])
            print()

            tweet_collection.insert(datajson)
            self.counter += 1
            if self.counter < self.limit:
                return True
            else:
                streamer.disconnect()

        except Exception as e:
            print(e)

if __name__=='__main__':

    listener = StdOutListener()
    streamer = tweepy.Stream(auth=auth, listener=listener)

    WORDS = ['boris johnson', 'jeremy corbyn']
    LONDON_COORDS =[-0.489, 51.28, 0.236, 51.686]

    streamer.filter(track=WORDS, locations=LONDON_COORDS, languages=['en'])

    cursor = tweet_collection.find()
    print(cursor.count())
