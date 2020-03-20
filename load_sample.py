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
    db = client.tweet_db2
    streaming_tweets = db.streaming_tweets
    rest_tweets = db.rest_tweets
    print('Database created')

    cursor1 = list(streaming_tweets.find())

    #  number to collect for 10 minute samples
    num_stream = round(len(cursor1)/6)

    stream_file = open("data/sample_tweets.json", "w")
    stream_file.write('[')

    for document in cursor1[:int(num_stream)]:
        stringdoc = json.dumps(document, cls=JSONEncoder)
        stream_file.write(stringdoc)
        stream_file.write(',\n')
    stream_file.write(']')

    print(num_stream, 'Stream tweets saved ')


def load_sample_stream_tweets():

    with open('data/sample_tweets.json') as f:
        tweetdb = json.load(f)

    return tweetdb






if __name__=='__main__':

    mongo_to_csv()
