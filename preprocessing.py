import json
import os
import re
from itertools import count
import pandas as pd


def get_user_data(tweets_df: pd.DataFrame) -> pd.DataFrame:
    all_users = tweets_df['user'].drop_duplicates()
    all_users_df = pd.DataFrame(all_users.to_list())[
        ['id', 'followers_count', 'verified', 'protected', 'description', 'friends_count', 'listed_count',
         'default_profile_image', 'default_profile', 'statuses_count', 'favourites_count', 'created_at']]
    all_users_df = all_users_df.groupby(['id'], as_index=False).aggregate('max')
    all_users_df['created_at'] = all_users_df['created_at'].apply(
        lambda x: pd.to_datetime(x, format='%a %b %d %H:%M:%S %z %Y'))
    return all_users_df


def extract_news_data(article_dir_path: str, article_dir_name: str):
    news_content_path = os.path.join(article_dir_path, 'news content.json')
    with open(news_content_path, 'r') as news_content_file:
        news_content = json.load(news_content_file)
        if not news_content['text'].isspace():
            return {'article_dir': article_dir_name,
                    'content_text': news_content['text'],
                    'n_images': len(news_content['images']),
                    'title': news_content['title'],
                    }
        return {}


def get_all_retweets(tweets_df):
    tweet_retweet_tweet = tweets_df[['retweeted_status', 'id']].explode('retweeted_status').dropna().rename(
        mapper={'retweeted_status': 'retweeted_id', 'id': 'tweet_id'}, axis='columns')

    return tweets_df, tweet_retweet_tweet


# def get_retweets_data(articles_directory) -> pd.DataFrame:
# retweets_data = pd.DataFrame(columns=['article_dir', 'tweet_id'])
# for single_article_dir in os.listdir(articles_directory):
#     try:
#         single_article_dir_path = os.path.join(articles_directory, single_article_dir)
#         if single_article_dir[0] == '.':
#             continue
#         if not 'news content.json' in os.listdir(single_article_dir_path):
#             continue
#         retweets_dir_path = os.path.join(single_article_dir_path, 'retweets')
#         for retweet_file in os.listdir(retweets_dir_path):
#             new_row = pd.DataFrame({'article_dir': [single_article_dir], 'tweet_id': [retweet_file[:-5]]})
#             retweets_data = pd.concat([retweets_data, new_row], ignore_index=True)
#
#     except FileNotFoundError:
#         pass
# return retweets_data
def get_articles_data(articles_directories: dict) -> pd.DataFrame:
    articles_data = []
    print("start")
    for articles_directory, label in articles_directories.items():
        j = 0
        for i, single_article_dir in enumerate(os.listdir(articles_directory)):
            j += 1

            single_article_dir_path = os.path.join(articles_directory, single_article_dir)
            if not i%100:
                print(f'article {i} - {single_article_dir_path}')
            try:
                news_data = extract_news_data(single_article_dir_path, single_article_dir)
                news_data['label'] = label
                if news_data:
                    articles_data.append(news_data)
            except FileNotFoundError:
                pass
            if j == 500:
                break
    articles_data = pd.DataFrame(articles_data).drop_duplicates()
    articles_data.to_csv('raw_articles_data.csv')
    return articles_data


def get_tweets_data(articles_directories) -> pd.DataFrame:
    tweets_data = []
    for articles_directory in articles_directories:
        i = 0
        for single_article_dir in os.listdir(articles_directory):
            try:
                single_article_dir_path = os.path.join(articles_directory, single_article_dir)
                tweets_dir_path = os.path.join(single_article_dir_path, 'tweets')
                for tweet_file in os.listdir(tweets_dir_path):
                    tweet_path = os.path.join(tweets_dir_path, tweet_file)
                    single_file_tweet_data = extract_tweet_data(tweet_path, single_article_dir)
                    if single_file_tweet_data:
                        tweets_data.append(single_file_tweet_data)
                i += 1
                if i == 500:
                    break
            except FileNotFoundError:
                pass
        return pd.DataFrame(tweets_data).drop_duplicates('id')


def extract_tweet_data(tweet_path: str, article_dir_name: str):
    with open(tweet_path, 'r') as tweet_file:
        raw_data = json.load(tweet_file)
        user = raw_data['user']
        created_at = pd.to_datetime(raw_data['created_at'], format='%a %b %d %H:%M:%S %z %Y')

        pattern = r'>(.*?)</a>'
        source = re.search(pattern, raw_data['source']).group(1)

        return {'article_dir': article_dir_name,
                'id': raw_data['id'],
                'user': user,
                'user_id': user['id'],
                'created_at': created_at,
                'source': source,
                'truncated': raw_data['truncated'],
                'retweet_count': raw_data['retweet_count'],
                'favorite_count': raw_data['favorite_count'],
                'favorited': raw_data['favorited'],
                'retweeted': raw_data['retweeted'],
                'in_reply_to_status_id_str': raw_data['in_reply_to_status_id_str'],
                'is_quote_status': raw_data['is_quote_status'],
                'hashtags': raw_data['entities']['hashtags'],
                'urls': raw_data['entities']['urls'],
                'user_mentions': [item['id_str'].lower() for item in raw_data['entities']['user_mentions']],
                "retweeted_status": raw_data.get('retweeted_status', {}).get('id')
                }


def get_all_hashtags(tweets_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    def generate_unique_id():
        counter = count(start=1)
        while True:
            yield next(counter)

    tweets_df['hashtags'] = tweets_df['hashtags'].apply(lambda x: [item['text'].lower() for item in x])
    id_generator = generate_unique_id()
    string_id_dict = {string: next(id_generator) for string in tweets_df['hashtags'].apply(pd.Series).stack().unique()}

    tweets_df['hashtags'] = tweets_df['hashtags'].apply(lambda lst: [string_id_dict[string] for string in lst])
    hashtag_df = pd.DataFrame(list(string_id_dict.items()), columns=['hashtag', 'id'])
    hashtag_tweet_df = tweets_df[['hashtags', 'id']].explode('hashtags').dropna().rename(
        mapper={'hashtags': 'hashtag_id', 'id': 'tweet_id'}, axis='columns')

    return hashtag_df, tweets_df, hashtag_tweet_df


def get_all_user_mentions(tweets_df: pd.DataFrame, users: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    user_mentions_tweet_df = tweets_df[['user_mentions', 'id']].explode('user_mentions').dropna().rename(
        mapper={'user_mentions': 'user_id', 'id': 'tweet_id'}, axis='columns')

    user_ids = users['id'].unique()
    user_mentions_tweet_df = user_mentions_tweet_df[user_mentions_tweet_df['user_id'].isin(user_ids)]

    return tweets_df, user_mentions_tweet_df


def get_all_links(tweets_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    def generate_unique_id():
        counter = count(start=1)
        while True:
            yield next(counter)

    tweets_df['urls'] = tweets_df['urls'].apply(lambda x: [item['url'].lower() for item in x])
    id_generator = generate_unique_id()
    string_id_dict = {string: next(id_generator) for string in tweets_df['urls'].apply(pd.Series).stack().unique()}

    tweets_df['urls'] = tweets_df['urls'].apply(lambda lst: [string_id_dict[string] for string in lst])
    urls_df = pd.DataFrame(list(string_id_dict.items()), columns=['url', 'id'])
    url_tweet_df = tweets_df[['urls', 'id']].explode('urls').dropna()
    url_tweet_df.rename(mapper={'urls': 'url_id', 'id': 'tweet_id'}, axis='columns', inplace=True)

    return urls_df, tweets_df, url_tweet_df


def get_all_sources(tweets_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    def generate_unique_id():
        counter = count(start=1)
        while True:
            yield next(counter)

    tweets_df['source'] = tweets_df['source'].apply(lambda x: x.lower())
    id_generator = generate_unique_id()
    string_id_dict = {string: next(id_generator) for string in tweets_df['source'].apply(pd.Series).stack().unique()}

    tweets_df['source'] = tweets_df['source'].apply(lambda source: string_id_dict[source])
    urls_df = pd.DataFrame(list(string_id_dict.items()), columns=['source', 'id'])
    url_tweet_df = tweets_df[['source', 'id']].explode('source').dropna()
    url_tweet_df.rename(mapper={'source': 'source_id', 'id': 'tweet_id'}, axis='columns', inplace=True)

    return urls_df, tweets_df, url_tweet_df


def load_all_csv_files(directory_paths: dict):
    articles_df = get_articles_data(directory_paths)
    tweets = get_tweets_data(directory_paths.keys())
    tweets = tweets[tweets['article_dir'].isin(articles_df['article_dir'])]
    # retweets = get_retweets_data(directory_path)
    users = get_user_data(tweets)
    tweets.drop(columns=['user'], inplace=True)

    # prepare user_tweet relation
    user_tweet = tweets[['id', 'user_id']].rename({'id': 'tweet_id'}, axis='columns')

    # prepare tweet_retweet relation
    tweet_reply_tweet = tweets[['id', 'in_reply_to_status_id_str']].rename(
        mapper={'id': 'reply_tweet_id', 'in_reply_to_status_id_str': 'source_tweet_id'},
        axis='columns')
    tweet_reply_tweet = tweet_reply_tweet.dropna()
    tweet_reply_tweet = tweet_reply_tweet[tweet_reply_tweet['source_tweet_id'].isin(tweets['id'])]
    # prepare tweet_hashtag relation

    hashtag_df, tweets, hashtag_tweet_df = get_all_hashtags(tweets)

    # prepare tweet_url relation

    url_df, tweets, url_tweet_df = get_all_links(tweets)

    # prepare tweet_source relation

    source_df, tweets, source_tweet_df = get_all_sources(tweets)

    # prepare user_mention_tweet relation

    tweets, user_mention_tweet_df = get_all_user_mentions(tweets, users)

    tweets, tweets_retweets_df = get_all_retweets(tweets)

    if not os.path.exists(os.path.join('results')):
        os.makedirs(os.path.join('results'))

    source_df.to_csv(os.path.join('results', 'sources.csv'), index=False)
    user_mention_tweet_df.to_csv(os.path.join('results', 'user_mention.csv'), index=False)
    tweets.to_csv(os.path.join('results', 'tweets.csv'), index=False)
    source_tweet_df.to_csv(os.path.join('results', 'source_tweet.csv'), index=False)
    url_tweet_df.to_csv(os.path.join('results', 'url_tweet_df.csv'), index=False)
    url_df.to_csv(os.path.join('results', 'url_df.csv'), index=False)
    hashtag_df.to_csv(os.path.join('results', 'hashtag_df.csv'), index=False)
    hashtag_tweet_df.to_csv(os.path.join('results', 'hashtag_tweet_df.csv'), index=False)
    tweet_reply_tweet.to_csv(os.path.join('results', 'tweet_reply_tweet.csv'), index=False)
    user_tweet.to_csv(os.path.join('results', 'user_tweet.csv'), index=False)
    articles_df.to_csv(os.path.join('results', 'articles_df.csv'), index=False)
    users.to_csv(os.path.join('results', 'users.csv'), index=False)
    tweets_retweets_df.to_csv(os.path.join('results', 'retweets.csv'), index=False)


if __name__ == '__main__':
    load_all_csv_files({'C:\\Users\\Hubert\\Downloads\\code\\code\\fakenewsnet_dataset\\gossipcop\\real': True,
                        'C:\\Users\\Hubert\\Downloads\\code\\code\\fakenewsnet_dataset\\gossipcop\\fake': False,
                        'C:\\Users\\Hubert\\Downloads\\code\\code\\fakenewsnet_dataset\\politifact\\real': True,
                        'C:\\Users\\Hubert\\Downloads\\code\\code\\fakenewsnet_dataset\\politifact\\fake': False,
                        })
 # load_all_csv_files({os.path.join('gossippart', 'real'): True,
    #                     # os.path.join('politifact', 'real'): True,
    #                     os.path.join('gossippart', 'fake'): False,
    #                     # os.path.join('politifact', 'fake'): False
    #                     })
