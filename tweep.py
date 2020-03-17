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
hashtags = []
rate_lim = False
rate_lim2 = False
auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

usersli = api.create_list('users')
faveli = []
ids = []
timeline_counter = 0
list_counter = 0

def timer(var):
    print('timer')
    global rate_lim, rate_lim2
    time.sleep(60*15)
    print('timerdone')
    if var == 1:
        rate_lim = True
    if var == 2:
        rate_lim2 = True



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
    #hashtags = Thread(target=datamanip, args=(datajason,))
    #hashtags.start()
    if rate_lim == False:
        timeline(datajason['user'])
    elif rate_lim2 == False:
        userlist_timeline()

def datamanip(datajason):
    print()

def timeline(user):
    if user['protected'] is False and user['statuses_count'] > 300:
        for page in timeline_limit(Cursor(api.user_timeline, user_id=user['id'], tweet_mode="extended", count=200).pages(1)):
            for tweet in page:
                rest_tweets.insert_one(tweet._json)


def timeline_limit(cursor):
    while True:
        try:
            yield cursor.next()
        except Exception as e:
            print('Ratelim1')
            global rate_lim
            rate_lim = True
            return


def adduserlist(user):
    global usersli
    usersli = api.add_list_member(user_id=user['id'], list_id=usersli.id)


def userlist_timeline():
    print('1')
    for page in userlist_limit(Cursor(api.list_timeline, list_id=usersli.id, count=200).pages()):
        for tweet in page:
            rest_tweets.insert_one(tweet._json)


def userlist_limit(cursor):
    while True:
        try:
            yield cursor.next()
        except Exception as e:
            print('Ratelim2')
            global rate_lim2
            rate_lim2 = True
            return


def search():
        for tweet in Cursor(api.search, geocode="51.5074,0.1278,30km", lang="en", include_entities=True).items(100):
            rest_tweets.insert_one(tweet._json)
            adduserlist(tweet._json['user'])



def start_streamer():
    listener = MyStreamListener()
    streamer = tweepy.Stream(auth=auth, listener=listener, tweet_mode='extended')
    streamer.filter(track=fronteir, locations=LONDON_COORDS, languages=['en'], stall_warnings=True, is_async=True)


#start_streamer()
search()
userlist_timeline()
d = api.rate_limit_status()
print(d['resources'])


