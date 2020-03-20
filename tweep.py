import re
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

t1 = True
t2 = False
t3 = False

li10 = []
userids = []
auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = API(auth, wait_on_rate_limit_notify=True,)
SEARCHED_HASHTAGS = []
OVERFLOW_HASHTAGS = []
TO_SEARCH = []

ids = []
timeline_counter = 0
userfave_counter = 0
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
            datamanip(datajson)
            self.q.task_done()


def datamanip(datajson):
    global timeline_counter, userfave_counter, SEARCHED_HASHTAGS, TO_SEARCH, OVERFLOW_HASHTAGS
    try:
        hasht = datajson['entities']['hashtags']
        if len(hasht) > 0:
            for tag in hasht:
                if tag not in SEARCHED_HASHTAGS:
                    if rate_lim3 == False:
                            TO_SEARCH.append(tag['text'])
                            if len(TO_SEARCH) == 5:
                                Thread(target=search_hashtags, args=(tag,)).start()
                                SEARCHED_HASHTAGS = SEARCHED_HASHTAGS + TO_SEARCH
                                TO_SEARCH = []
                    else:
                        OVERFLOW_HASHTAGS.append(tag['text'])
    except:
        return
    if datajson['user']['protected'] is False and datajson['user']['statuses_count'] > 3200 and datajson['user']['favourites_count']  > 100:
        user = datajson['user']['screen_name']
        if rate_lim == False and timeline_counter < 3:
            timeline_counter = timeline_counter + 1
            Thread(target=get_timeline, args=(user,)).start()
            return
        elif rate_lim2 == False and userfave_counter < 3:
            userfave_counter = userfave_counter + 1
            Thread(target=get_faves, args=(user,)).start()
            return
    else:
        return


def get_timeline(user):
    global timeline_counter
    for page in timeline_limit(Cursor(api.user_timeline, id=user, tweet_mode="extended", count=200).pages(16)):
        for tweet in page:
            rest_tweets.insert_one(tweet._json)
    timeline_counter = timeline_counter - 1



def timeline_limit(cursor):
    try:
        yield cursor.next()
    except StopIteration:
        return
    except tweepy.TweepError as e:
        print(e.args[0])
        global rate_lim
        rate_lim = True
    return


def get_faves(user):
    global userfave_counter
    for tweet in fave_limit(Cursor(api.favorites, id=user).items(100)):
        rest_tweets.insert_one(tweet._json)
    userfave_counter = userfave_counter - 1



def fave_limit(cursor):
    while True:
        try:
            yield cursor.next()
        except StopIteration:
            return
        except tweepy.TweepError as e:
                print(e.response.text)
                global rate_lim2
                rate_lim2 = True
        return



def search_hashtags(hlist):
        for tweet in search_limit(Cursor(api.search, q=hlist, geocode="51.5074,0.1278,30km", lang="en", include_entities=True).items(400)):
            rest_tweets.insert_one(tweet._json)

def search_limit(cursor):
    while True:
        try:
            yield cursor.next()
        except StopIteration:
            return
        except tweepy.TweepError as e:
            print(e.response.text)
            global rate_lim3
            rate_lim3 = True
        return


def start_streamer():
    listener = MyStreamListener()
    streamer = tweepy.Stream(auth=auth, listener=listener, tweet_mode='extended')
    streamer.filter(track=fronteir, locations=LONDON_COORDS, languages=['en'], stall_warnings=True, is_async=True)


d = api.rate_limit_status()
print(d['resources']['favorites']['/favorites/list'])
print(d['resources']['search']['/search/tweets'])
print(d['resources']['statuses']['/statuses/user_timeline'])

start_streamer()
t = 0

while t != 60:
    time.sleep(60 * 1)
    d = api.rate_limit_status()
    faver = d['resources']['favorites']['/favorites/list']['remaining']
    search = d['resources']['search']['/search/tweets']['remaining']
    timeline = d['resources']['statuses']['/statuses/user_timeline']['remaining']
    print(faver,search, timeline)
    if timeline == 900:
        rate_lim = False
        timeline_counter = 0
    if search == 180:
        rate_lim3 = False
        Thread(target=search_hashtags, args=(OVERFLOW_HASHTAGS,)).start()
        SEARCHED_HASHTAGS = SEARCHED_HASHTAGS + OVERFLOW_HASHTAGS
    if faver == 75:
        rate_lim2 = False
        userfave_counter = 0
    t += 1
    print(rate_lim, rate_lim2, rate_lim3, timeline_counter, userfave_counter)


