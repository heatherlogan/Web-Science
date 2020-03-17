import sys
import time
import threading
from queue import Queue
from threading import Thread
from tweepy.auth import OAuthHandler
from tweepy import API, Cursor
import tweepy
from pymongo import MongoClient
import json


client=MongoClient()
db=client.tweet_db
streaming_tweets = db.streaming_tweets
rest_tweets = db.rest_tweets
print('Database created')


CONSUMER_KEY = "TYPVBk4kO17UMfDZSOURAGL4E"
CONSUMER_SECRET = "0Kq5zHsOb4lNfDGGbwOmhVaXTkpwS1DZiA5kBgpFt6L3OilJdt"
ACCESS_TOKEN = "431764921-CaJPtXrUEu6PoYsyYozOFMrAu7rbSBZKUOppVpBq"
ACCESS_TOKEN_SECRET = "N1qZUsVN5h4vz31MdIAwJWIkO6oJCJUBx6PSlIXRvKqSj"
LONDON_COORDS = [-0.489, 51.48, -0.18, 51.686]
counter = 0
fronteir = ['borisjohnson',
            'jeremycorbyn',
            'theresamay',
            'tories',
            'labour',
            'conservatives',
            'uk',
            'britain',
            'british politics',
            'ukpolitics',
            'brexit',
            'europe',
            'european union',
            'ukip',
            'prime minister']

rate_lim = False
rate_lim2 = False
rate_lim3 = False

userids = []

auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = API(auth, wait_on_rate_limit_notify=True)
SEARCHED_HASHTAGS = []
OVERFLOW_HASHTAGS = []
ids = []
timeline_counter = 0
list_counter = 0
name = 0
var = 1



class MyStreamListener(tweepy.StreamListener):
    def __init__(self, q=Queue()):
        super().__init__()
        num_worker_threads = 3
        self.q = q
        for i in range(num_worker_threads):
            t = Thread(target=self.handle_data)
            t.daemon = True
            t.start()

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
            schedular(datajson)
            self.q.task_done()


def schedular(datajason):
    Thread(target=datamanip, args=(datajason,)).start()
    if rate_lim == False:
        #Thread(target=timeline, args=(datajason['user'],)).start()
        #timeline(datajason['user'])
        return
    elif rate_lim2 == False:
        #Thread(target=faves, args=(datajason['user'],)).start()
        #faves(datajason['user'])
        return
    else:
        return


def datamanip(datajson):
    try:
        hasht = datajson['entities']['hashtags']
        if len(hasht) > 0:
            for tag in hasht:
                if tag not in SEARCHED_HASHTAGS:
                    if rate_lim4 == False:
                        SEARCHED_HASHTAGS.append(tag['text'])
                        Thread(target=search_hashtags, args=(tag,)).start()
                    else:
                        OVERFLOW_HASHTAGS.append(tag['text'])
        return
    except:
        return


def timeline(user):
    if user['protected'] is False and user['statuses_count'] > 3200:
        for page in timeline_limit(Cursor(api.user_timeline, user_id=user['id'], tweet_mode="extended", count=200).pages(16)):
            for tweet in page:
                rest_tweets.insert_one(tweet._json)
    else:
        return


def timeline_limit(cursor):
    try:
        yield cursor.next()
    except StopIteration:
        return
    except tweepy.TweepError as e:
        print('User Timeline limit')
        global rate_lim
        rate_lim = True
    return



def faves(user):
    if user['favourites_count'] > 20:
        for tweet in fave_limit(Cursor(api.favorites, id=user['id']).items(20)):
            rest_tweets.insert_one(tweet._json)
    else:
        return


def fave_limit(cursor):
    while True:
        try:
            yield cursor.next()
        except StopIteration:
            return
        except tweepy.TweepError as e:
            print('User fave limit')
            global rate_lim2
            rate_lim2 = True
        return



def search_hashtags(hlist):
        for tweet in search_limit(Cursor(api.search, q=hlist, geocode="51.5074,0.1278,30km", lang="en", include_entities=True).items(100)):
            rest_tweets.insert_one(tweet._json)


def search_limit(cursor):
    while True:
        try:
            yield cursor.next()
        except StopIteration:
            return
        except tweepy.TweepError as e:
            print('Search Limit')
            global rate_lim4
            rate_lim4 = True
        return


def start_streamer():
    listener = MyStreamListener()
    streamer = tweepy.Stream(auth=auth, listener=listener, tweet_mode='extended')
    streamer.filter(track=fronteir, locations=LONDON_COORDS, languages=['en'], stall_warnings=True, is_async=True)


d = api.rate_limit_status()
print(d['resources']['favorites'])
print(d['resources']['lists'])
print(d['resources']['search'])
print(d['resources']['statuses'])


start_streamer()
t = 0
while t != 4:
    time.sleep(60 * 15)
    print(threading.enumerate())
    print(d['resources']['favorites'])
    print(d['resources']['lists'])
    print(d['resources']['search'])
    print(d['resources']['statuses'])
    t += 1
    rate_lim = False
    rate_lim3 = False
    rate_lim2 = False
    Thread(target=search_hashtags, args=(OVERFLOW_HASHTAGS,)).start()
    SEARCHED_HASHTAGS = SEARCHED_HASHTAGS + OVERFLOW_HASHTAGS
print(d['resources']['favorites'])


