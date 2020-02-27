import time
import threading
import tweepy
from tweepy import API, Cursor, OAuthHandler
import json
import pymongo
from pymongo import MongoClient
from threading import Thread

# TWITTER API Authentication

CONSUMER_KEY = "TYPVBk4kO17UMfDZSOURAGL4E"
CONSUMER_SECRET = "0Kq5zHsOb4lNfDGGbwOmhVaXTkpwS1DZiA5kBgpFt6L3OilJdt"
ACCESS_TOKEN = "431764921-CaJPtXrUEu6PoYsyYozOFMrAu7rbSBZKUOppVpBq"
ACCESS_TOKEN_SECRET = "N1qZUsVN5h4vz31MdIAwJWIkO6oJCJUBx6PSlIXRvKqSj"

SEARCHED_HASHTAGS = []
COUNTER = 0


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#restAPI

#wait on rate limit avoids time out
api = tweepy.API(auth)
# Database Setup
client=MongoClient()
db=client.tweet_db
streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets
print('Database created')


class TwitterClient():
    def __init__(self, twitter_user=None, hashtag=None):
        self.auth = TwitterAuthenticator.authenticate_twitter_app(self)
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user
        self.hashtag = hashtag

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

    def get_hashtag_tweets(self, n):
        tweets = []
        for tweet in tweepy.Cursor(api.search, q=self.hashtag, lang='en').items(n):
            tweets.append(tweet)
        return tweets

class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        self.auth = OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15*60)

def crawl_users_tweets(user):
    # print('crawling', user)
    twitter_client = TwitterClient(twitter_user=user)
    try:
        user_tl_tweets = twitter_client.get_user_timeline_tweets(50)
        for tweet in user_tl_tweets:
            rest_tweets.insert_one(tweet._json)
    except:
        pass

def crawl_hashtags(hashtag):
    # print('crawling ', hashtag)
    twitter_client = TwitterClient(hashtag=hashtag)
    try:
        hashtag_tweets = twitter_client.get_hashtag_tweets(50)
        for tweet in hashtag_tweets:
            rest_tweets.insert_one(tweet._json)
    except:
        pass

class TwitterStreamListener(tweepy.StreamListener):

    def __init__(self, num, search_terms):
        super().__init__()
        self.counter = 0
        self.limit = 100
        self.num = num
        self.search_terms = search_terms
        self.searched_hashtags = []

        print('thread {} running ... '.format(num))

    def on_connect(self):
        print("Connected")

    def on_error(self, status_code):
        print('Error: ' + repr(status_code))
        return False

    def on_data(self, data):

        try:
            datajson = json.loads(data)
            user = datajson['user']['screen_name']

            # # checks for geo data in london
            # if datajson['geo'] is not None:
            #     threading.active_count()
            #     print(datajson['user']['screen_name'])
            #     print(datajson['geo'])
            #     print(user)
            #     thread = Thread(target=get_geo, args=[datajson])
            #     thread.start()

            # add hashtags to content based fronteir
            hashtags = datajson['entities']['hashtags']
            if len(hashtags)>0:
                for tag in hashtags:
                    self.search_terms.append(tag['text'])

            streaming_tweets.insert_one(datajson)

            # print(datajson['text']) the ID of the user. Helpful for disambiguating when a valid user ID is also a valid

            thread = threading.Thread(target=crawl_users_tweets, args=((user,)))
            thread.start()
            thread.join()

            for term in self.search_terms:
                if term not in SEARCHED_HASHTAGS:
                    SEARCHED_HASHTAGS.append(term)



        except Exception as e:
            pass



# def get_geo(datajson):
#     timeline = api.user_timeline(datajson['user']['screen_name'])
#     for tweet in timeline:
#         if tweet.geo is not None:
#             try:
#                 tweet_collection.insert_one(datajson)
#             except pymongo.errors.DuplicateKeyError:
#                 continue


def start_streamer(fronteir, num):

    listener = TwitterStreamListener(search_terms=fronteir, num=num)
    streamer = tweepy.Stream(auth=auth, listener=listener)
    streamer.filter(track=listener.search_terms, locations=LONDON_COORDS, languages=['en'])


if __name__=='__main__':

    LONDON_COORDS = [-0.489, 51.28, 0.236, 51.686]
    COUNTER = 0
    # tweet_collection.remove()


    print('STREAMING TWEETS ', streaming_tweets.count())
    print('REST TWEETS ', rest_tweets.count())

    start = time.perf_counter()

    fronteir = ['borisjohnson', 'jeremycorbyn', 'theresamay', 'tories',
                'labour', 'conservatives']
    fronteir2 = ['uk', 'britain', 'british politics', 'ukpolitics ']
    fronteir3 = ['brexit', 'europe', 'european union', 'ukip', 'prime minister']


    thread1 = threading.Thread(target=start_streamer, args=((fronteir, 1,)))
    thread1.start()

    thread2 = threading.Thread(target=start_streamer, args=((fronteir2, 2,)))
    thread2.start()

    thread3 = threading.Thread(target=start_streamer, args=((fronteir3, 3,)))
    thread3.start()

    time.sleep(60*50)
    finish=time.perf_counter()
    thread1.join()
    thread2.join()
    thread3.join()







