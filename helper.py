import pandas as pd

# helper to save task results

def tokenize(tweet):

    tweet = str(tweet).strip().split()
    return ' '.join(tweet)


if __name__=='__main__':

    pp_tweets = pd.read_csv('data/preprocessed_tweets.csv')

    a_results = pd.read_csv('data/task3/tweets/subtask_A_tweets.csv')
    b_results = pd.read_csv('data/task3/tweets/subtask_B_tweets.csv')
    b_results = b_results.set_index('idx')
    b_results = b_results.drop_duplicates('text')
    c_results = pd.read_csv('data/task3/tweets/subtask_C_tweets.csv')
    c_results = c_results.set_index('idx')

    pp_tweets['text'] = pp_tweets['text'].apply(tokenize)
    b_results['text'] = b_results['text'].apply(tokenize)
    c_results['text'] = c_results['text'].apply(tokenize)


    merged1 = pd.merge(pp_tweets, b_results, on='text')
    merged2 = pd.merge(merged1, c_results, on='text')
    merged2 = merged2.drop_duplicates('text')
    merged2 = merged2.drop('text', axis=1)

    full_tweets = pd.merge(a_results, merged2, on='tweet_id')
    full_tweets = full_tweets.drop_duplicates('tweet_id')
    #
    full_tweets.to_csv('data/task3/tweets/B_C_subtasks.csv')