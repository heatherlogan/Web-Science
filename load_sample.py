import json
import pymongo
from json import JSONEncoder

from bson import ObjectId
from pymongo import MongoClient


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

def mongo_to_csv():

    client = MongoClient()
    db = client.tweet_db
    streaming_tweets = db.streaming_tweets
    #rest_tweets = db.rest_tweets
    print('Database created')

    cursor1 = streaming_tweets.find()
    #cursor2 = streaming_tweets.find()

    cursor = list(cursor1)
    #cursor = list(cursor) + list(cursor2)

    file = open("sample_tweets.json", "w")
    file.write('[')

    for document in cursor:
        stringdoc = json.dumps(document, cls=JSONEncoder)
        file.write(stringdoc)
        file.write(',\n')
    file.write(']')



def load_csv_to_mongo():

    with open('sample_tweets.json') as f:
        tweetdb = json.load(f)

    return tweetdb


if __name__=='__main__':

    mongo_to_csv()
