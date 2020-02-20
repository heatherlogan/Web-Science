import tweepy
from tweepy import API, Cursor, OAuthHandler
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

class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator.authenticate_twitter_app(self)
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_user_timeline_tweets(self, n):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(n):
            tweets.append(tweet)
        return tweets

    def get_followers(self, n):
        follower_list = []
        for follower in Cursor(self.twitter_client.followers_ids, id=self.twitter_user).items(n):
            follower_list.append(follower)
        return follower_list



class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        self.auth = OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth


class TwitterStreamListener(tweepy.StreamListener):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.limit = 2

    def on_connect(self):
        print("Connected")


    def on_error(self, status_code):
        print('Error: ' + repr(status_code))
        return False

    def on_data(self, data):
        try:
            datajson = json.loads(data)
            # print(json.dumps(datajson, indent=4, sort_keys=True))

            user = datajson['user']['id']
            users.append(user)

            tweet_text = datajson['text']
            print(user, tweet_text)

            # add hashtags to content based fronteir
            hashtags = datajson['entities']['hashtags']
            if len(hashtags)>0:
                for tag in hashtags:
                    topic_fronteir.append(tag['text'])

            # print()

            tweet_collection.insert(datajson)
            self.counter += 1
            if self.counter < self.limit:
                return True
            else:
                streamer.disconnect()

        except Exception as e:
            print(e)


def crawl_users_tweets(user_list):

    new_followers = []

    for user in user_list:

        twitter_client = TwitterClient(user)
        users_followers = twitter_client.get_followers(2)
        user_tl_tweets = twitter_client.get_user_timeline_tweets(1)

        for follower in (users_followers):
            print(follower)





if __name__=='__main__':

    # users and topics to spawn crawler after initial stream
    topic_fronteir = []
    users = []

    listener = TwitterStreamListener()
    streamer = tweepy.Stream(auth=auth, listener=listener)

    WORDS = ['boris johnson', 'jeremy corbyn']
    LONDON_COORDS =[-0.489, 51.28, 0.236, 51.686]

    streamer.filter(track=WORDS, locations=LONDON_COORDS, languages=['en'])

    cursor = tweet_collection.find()
    print(cursor.count())

    print(topic_fronteir)
    print()
    print(users)

    crawl_users_tweets(users)

