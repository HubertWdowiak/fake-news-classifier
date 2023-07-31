import json
import os

import pandas as pd


def clean(articles_directories) -> pd.DataFrame:
    tweets_data = []
    for article_directory in articles_directories:
        for single_article_dir in os.listdir(article_directory):
            try:
                tweet_files = set()
                single_article_dir_path = os.path.join(article_directory, single_article_dir)
                tweets_dir_path = os.path.join(single_article_dir_path, 'tweets')
                retweets_dir_path = os.path.join(single_article_dir_path, 'retweets')
                for tweet_file in os.listdir(tweets_dir_path):
                    tweet_files.add(tweet_file)

                for retweet_file in os.listdir(retweets_dir_path):
                    print(article_directory, single_article_dir, retweet_file)
                    with open(os.path.join(retweets_dir_path, retweet_file), 'r') as file:
                        try:
                            content = json.load(file)
                        except Exception:
                            content = {'retweets': []}
                        is_empty = content == {'retweets': []}

                    if is_empty:
                        os.remove(os.path.join(retweets_dir_path, retweet_file))
                        continue
                    not_in_tweets_dir = retweet_file not in tweet_files
                    if not_in_tweets_dir:
                        os.remove(os.path.join(retweets_dir_path, retweet_file))

            except FileNotFoundError:
                pass
    return pd.DataFrame(tweets_data).drop_duplicates('id')


clean([
       os.path.join('politifact', 'real'),
       os.path.join('politifact', 'fake'),
       os.path.join('gossipcop', 'real'),
       os.path.join('gossipcop', 'fake')])
