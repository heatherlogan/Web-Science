import time
import threading
from queue import Queue
import tweepy
from tweepy import API, Cursor, AppAuthHandler
import json
from pymongo import MongoClient
from threading import Thread

# TWITTER API Authentication
CONSUMER_KEY = "TYPVBk4kO17UMfDZSOURAGL4E"
CONSUMER_SECRET = "0Kq5zHsOb4lNfDGGbwOmhVaXTkpwS1DZiA5kBgpFt6L3OilJdt"

ACCESS_TOKEN = "431764921-CaJPtXrUEu6PoYsyYozOFMrAu7rbSBZKUOppVpBq"
ACCESS_TOKEN_SECRET = "N1qZUsVN5h4vz31MdIAwJWIkO6oJCJUBx6PSlIXRvKqSj"

SEARCHED_HASHTAGS = []
COUNTER = 0

auth = AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
#auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#restAPI

api = tweepy.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True)

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
        self.auth = AppAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth


class TwitterStreamListener(tweepy.StreamListener):
    def __init__(self, num, search_terms, q = Queue()):
        super().__init__()
        self.counter = 0
        self.limit = 100
        self.num = num
        self.search_terms = search_terms
        self.searched_hashtags = []
        num_worker_threads = 3
        self.q = q
        for i in range(num_worker_threads):
            t = Thread(target=self.handle_data)
            t.daemon = True
            t.start()

        print('thread {} running ... '.format(num))

    def on_connect(self):
        print("Connected")

    def on_error(self, status_code):
        print('Error: ' + repr(status_code))
        return False

    def on_data(self, data):
        self.q.put(data)


    def handle_data(self):
        while True:
            data = self.q.get()
            datajson = json.loads(data)
            streaming_tweets.insert_one(datajson)
           # data_manip(self, datajson)
           # self.q.task_done()


def data_manip(self, datajson):
    try:
        user = datajson['user']['screen_name']
        hashtags = datajson['entities']['hashtags']
        if len(hashtags) > 0:
            for tag in hashtags:
                self.search_terms.append(tag['text'])
        thread = threading.Thread(target=crawl_users_tweets, args=((user,)))
        thread.start()
        thread.join()
        #thread.join blocks
        for term in self.search_terms:
            if term not in SEARCHED_HASHTAGS:
                SEARCHED_HASHTAGS.append(term)
                thread2 = threading.Thread(target=crawl_hashtags(term), args=((term,)))
                thread2.start()
    except Exception as e:
        pass



def crawl_users_tweets(user):
    twitter_client = TwitterClient(twitter_user=user)
    try:
        user_tl_tweets = twitter_client.get_user_timeline_tweets(100)
        for tweet in user_tl_tweets:
            rest_tweets.insert_one(tweet._json)
    except:
        pass



def crawl_hashtags(hashtag):

    twitter_client = TwitterClient(hashtag=hashtag)
    try:
        hashtag_tweets = twitter_client.get_hashtag_tweets(100)
        for tweet in hashtag_tweets:
            rest_tweets.insert_one(tweet._json)
    except:
        pass


def start_streamer(fronteir, num):
    listener = TwitterStreamListener(search_terms=fronteir, num=num)
    streamer = tweepy.Stream(auth=auth, listener=listener, tweet_mode='extended')
    streamer.filter(track=listener.search_terms, locations=LONDON_COORDS, languages=['en'], stall_warnings=True,)



if __name__=='__main__':

    LONDON_COORDS = [-0.489, 51.28, 0.236, 51.686]
    COUNTER = 0
    print('STREAMING TWEETS ', streaming_tweets.estimated_document_count())
    print('REST TWEETS ', rest_tweets.estimated_document_count())

    start = time.perf_counter()

    fronteir = ['borisjohnson', 'jeremycorbyn', 'theresamay', 'tories',
                'labour', 'conservatives']
    fronteir2 = ['uk', 'britain', 'british politics', 'ukpolitics ']
    fronteir3 = ['brexit', 'europe', 'european union', 'ukip', 'prime minister']

    allfront = fronteir + fronteir2 + fronteir3

    thread1 = threading.Thread(target=start_streamer, args=(allfront, 1,))
    thread1.start()
    print('started')

    time.sleep(60*60)
    finish=time.perf_counter()
    thread1.join()
    print('joined')






