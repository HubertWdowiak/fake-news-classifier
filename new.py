import json
import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch.optim import Adam
from torch.functional import F


def extract_news_data(article_dir_path: str, article_dir_name: str):
    news_content_path = os.path.join(article_dir_path, 'news content.json')
    with open(news_content_path, 'r') as news_content_file:
        news_content = json.load(news_content_file)
        if not news_content['text'].isspace():
            return {'article_dir': article_dir_name, 'content_text': news_content['text']}
        return {}


def extract_retweet_data(retweet_path: str, article_dir_name: str):
    with open(retweet_path, 'r') as retweet_file:
        retweets = []
        raw_data = json.load(retweet_file)
        for r in raw_data['retweets']:
            user = r['user']
            r.pop('user')
            created_at = pd.to_datetime(r['created_at'], format='%a %b %d %H:%M:%S %z %Y')
            retweets.append(
                {'article_dir': article_dir_name, 'retweet': r, 'user': user, 'created_at': created_at, 'id': r['id']})
        return retweets


def extract_tweet_data(tweet_path: str, article_dir_name: str):
    with open(tweet_path, 'r') as tweet_file:
        raw_data = json.load(tweet_file)
        user = raw_data['user']
        raw_data.pop('user')
        created_at = pd.to_datetime(raw_data['created_at'], format='%a %b %d %H:%M:%S %z %Y')
        return {'article_dir': article_dir_name, 'id': raw_data['id'], 'user': user, 'created_at': created_at}


def get_articles_data(articles_directory, label) -> pd.DataFrame:
    articles_data = []
    for single_article_dir in os.listdir(articles_directory):
        single_article_dir_path = os.path.join(articles_directory, single_article_dir)

        try:
            news_data = extract_news_data(single_article_dir_path, single_article_dir)
            news_data['label'] = label
            if news_data:
                articles_data.append(news_data)
        except FileNotFoundError:
            pass

    return pd.DataFrame(articles_data)


def get_retweets_data(articles_directory) -> pd.DataFrame:
    retweets_data = []
    for single_article_dir in os.listdir(articles_directory):
        try:
            single_article_dir_path = os.path.join(articles_directory, single_article_dir)
            retweets_dir_path = os.path.join(single_article_dir_path, 'retweets')
            for retweet_file in os.listdir(retweets_dir_path):
                retweet_path = os.path.join(retweets_dir_path, retweet_file)
                single_file_retweet_data = extract_retweet_data(retweet_path, single_article_dir_path)
                if single_file_retweet_data:
                    retweets_data.extend(single_file_retweet_data)
        except FileNotFoundError:
            pass
    return pd.DataFrame(retweets_data)


def get_tweets_data(articles_directory) -> pd.DataFrame:
    tweets_data = []
    for single_article_dir in os.listdir(articles_directory):
        try:
            single_article_dir_path = os.path.join(articles_directory, single_article_dir)
            tweets_dir_path = os.path.join(single_article_dir_path, 'tweets')
            for tweet_file in os.listdir(tweets_dir_path):
                tweet_path = os.path.join(tweets_dir_path, tweet_file)
                single_file_tweet_data = extract_tweet_data(tweet_path, single_article_dir)
                if single_file_tweet_data:
                    tweets_data.append(single_file_tweet_data)
        except FileNotFoundError:
            pass
    return pd.DataFrame(tweets_data)


def get_user_data(retweets_df, tweets_df) -> pd.DataFrame:
    all_users = list(pd.concat([retweets_df['user'], tweets_df['user']]).drop_duplicates())
    all_users_df = pd.DataFrame.from_dict(all_users)[
        ['id', 'followers_count', 'friends_count', 'listed_count', 'favourites_count', 'created_at']]
    all_users_df = all_users_df.groupby(['id'], as_index=False).aggregate('max')
    all_users_df['created_at'] = all_users_df['created_at'].apply(
        lambda x: pd.to_datetime(x, format='%a %b %d %H:%M:%S %z %Y'))
    return all_users_df


def load_node_csv(df, encoders=None, **kwargs):
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


class DateEncoder:
    def __call__(self, x):
        df = pd.DataFrame()
        df['year'] = x.apply(lambda a: pd.to_datetime(a).year)
        df['month'] = x.apply(lambda a: pd.to_datetime(a).month)
        df['day'] = x.apply(lambda a: pd.to_datetime(a).day)
        df['hour'] = x.apply(lambda a: pd.to_datetime(a).hour)
        df['minute'] = x.apply(lambda a: pd.to_datetime(a).minute)
        df['second'] = x.apply(lambda a: pd.to_datetime(a).second)
        return torch.tensor(df.values)


class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


if __name__ == '__main__':
    torch.manual_seed(42)

    # fake_directory = os.path.join('politfact', 'fake')
    # articles_df = get_articles_data(fake_directory, False)
    # tweets = get_tweets_data(fake_directory)
    # tweets = tweets[tweets['article_dir'].isin(articles_df['article_dir'])]
    # retweets = get_retweets_data(fake_directory)
    # users = get_user_data(retweets, tweets)
    #
    # tweets['user'] = tweets['user'].apply(lambda x: x['id'])
    # retweets['user'] = retweets['user'].apply(lambda x: x['id'])
    # user_tweet = tweets[['id', 'user']].rename({'id': 'tweet_id', 'user': 'user_id'}, axis='columns')
    # tweet_article = tweets[['article_dir', 'id']].rename({'id': 'tweet_id'}, axis='columns')
    # tweets.drop(columns=['user'], inplace=True)
    #
    # ###################################################
    #
    # real_directory = os.path.join('politfact', 'real')
    # articles_df_real = get_articles_data(real_directory, True)
    # tweets_real = get_tweets_data(real_directory)
    # tweets_real = tweets_real[tweets_real['article_dir'].isin(articles_df_real['article_dir'])]
    # retweets_real = get_retweets_data(real_directory)
    # if retweets_real.empty:
    #     retweets_real = pd.DataFrame([], columns=retweets.columns)
    # users_real = get_user_data(retweets_real, tweets_real)
    #
    # tweets_real['user'] = tweets_real['user'].apply(lambda x: x['id'])
    # retweets_real['user'] = retweets_real['user'].apply(lambda x: x['id'])
    # user_tweet_real = tweets_real[['id', 'user']].rename({'id': 'tweet_id', 'user': 'user_id'}, axis='columns')
    # tweet_article_real = tweets_real[['article_dir', 'id']].rename({'id': 'tweet_id'}, axis='columns')
    # tweets_real.drop(columns=['user'], inplace=True)
    #
    # pd.concat([tweets, tweets_real]).drop_duplicates().to_csv(os.path.join('extracted_data', 'tweets.csv'), index=False)
    # pd.concat([users, users_real]).drop_duplicates().to_csv(os.path.join('extracted_data', 'users.csv'), index=False)
    # pd.concat([articles_df, articles_df_real]).drop_duplicates().to_csv(os.path.join('extracted_data', 'articles.csv'),
    #                                                                     index=False)
    # pd.concat([user_tweet, user_tweet_real]).drop_duplicates().to_csv(os.path.join('extracted_data', 'user_tweet.csv'),
    #                                                                   index=False)
    # pd.concat([tweet_article, tweet_article_real]).drop_duplicates().to_csv(
    #     os.path.join('extracted_data', 'tweet_article.csv'), index=False)

    df = pd.read_csv(os.path.join('extracted_data', 'tweets.csv'), index_col='id').sample(frac=1, random_state=42)
    tweet_x, tweet_mapping = load_node_csv(df, index_col='id', encoders={'created_at': DateEncoder()})
    tweet_column_max = tweet_x.max(dim=0).values
    tweet_x = tweet_x / tweet_column_max

    df = pd.read_csv(os.path.join('extracted_data', 'users.csv'), index_col='id').sample(frac=1, random_state=42)
    user_x, user_mapping = load_node_csv(df, index_col='id', encoders={'created_at': DateEncoder()})
    user_column_max = user_x.max(dim=0).values
    user_x = user_x / user_column_max

    df = pd.read_csv(os.path.join('extracted_data', 'articles.csv'), index_col='article_dir').sample(frac=1,
                                                                                                     random_state=42)
    labels = df['label']
    df = df.drop(columns=['label'])
    article_x, article_mapping = load_node_csv(df, index_col='article_dir',
                                               encoders={'content_text': SequenceEncoder()})
    article_column_max = article_x.max(dim=0).values
    article_x = article_x / article_column_max

    data = HeteroData()
    data['tweet'].x = tweet_x
    data['article'].x = article_x
    data['user'].x = user_x

    data['article'].y = torch.tensor(labels, dtype=torch.float).unsqueeze(dim=-1)

    edge_index, edge_label = load_edge_csv(
        os.path.join('extracted_data', 'tweet_article.csv'),
        src_index_col='tweet_id',
        src_mapping=tweet_mapping,
        dst_index_col='article_dir',
        dst_mapping=article_mapping,
    )

    data['tweet', 'relates', 'article'].edge_index = edge_index
    data['tweet', 'relates', 'article'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('extracted_data', 'user_tweet.csv'),
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['user', 'creates', 'tweet'].edge_index = edge_index
    data['user', 'creates', 'tweet'].edge_label = edge_label

    data_idx = list(article_mapping.values())
    train_mask, test_mask = train_test_split(data_idx, random_state=42)


    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
            self.lin1 = Linear(-1, hidden_channels)
            self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
            self.lin2 = Linear(-1, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index) + self.lin1(x)
            x = x.relu()
            x = self.conv2(x, edge_index) + self.lin2(x)
            return torch.sigmoid(x)


    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)

    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=len(train_mask),
        input_nodes=('article', torch.tensor(train_mask))
    )

    model = GAT(hidden_channels=128, out_channels=1)
    model = to_hetero(model, data.metadata(), aggr='sum')
    # model.to('cuda:0')

    optimizer = Adam(model.parameters())


    def train(epochs: int = 1000):
        model.train()

        for epoch in range(epochs):
            total_examples = total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                # batch = batch.to('cuda:0')
                batch_size = batch['article'].batch_size
                out = model({k: v.float() for k, v in batch.x_dict.items()}, batch.edge_index_dict)
                loss = F.binary_cross_entropy(out['article'][:batch_size],
                                              batch['article'].y[:batch_size])
                loss.backward()
                optimizer.step()

                total_examples += batch_size
                total_loss += float(loss) * batch_size
            print(total_loss / total_examples)

        return total_loss / total_examples


    train()
