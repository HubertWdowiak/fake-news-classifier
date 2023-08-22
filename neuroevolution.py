import os
import torch_geometric.transforms as T

import pandas as pd
import pygad
import torch
from sklearn.model_selection import train_test_split
from torch.functional import F
from torch.optim import Adam
from torch.optim import SGD, RMSprop
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, Linear, to_hetero


from utils import load_node_csv, DateEncoder, DefaultEncoder, SequenceEncoder, load_edge_csv


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=heads, concat=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return torch.sigmoid(x)


def get_optimizer(idx):
    optimizers = [Adam, SGD, RMSprop]
    return optimizers[int(idx)]


def get_aggregation(idx):
    aggrs = ['sum', 'mean', 'max', 'min', 'mul']
    return aggrs[int(idx)]


def create_model(solution):
    optimizer_type = get_optimizer(solution[0])
    aggregation = get_aggregation(solution[1])
    learning_rate = int(solution[2])/1000
    heads = int(solution[3])
    neurons = int(solution[4])

    model = GAT(hidden_channels=neurons, out_channels=1, heads=heads)
    model = to_hetero(model, data.metadata(), aggr=aggregation)
    optimizer = optimizer_type(lr=learning_rate, params=model.parameters())
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model, optimizer


def evaluate(model, data_loader):
    model.eval()
    total_examples = total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            batch_size = batch['article'].batch_size
            out = model({k: v.float() for k, v in batch.x_dict.items()}, batch.edge_index_dict)
            loss = F.binary_cross_entropy(out['article'][:batch_size], batch['article'].y[:batch_size])

            total_examples += batch_size
            total_loss += float(loss) * batch_size

    return total_loss / total_examples


def train(model, train_loader, validation_loader, optimizer, epochs: int = 10):
    model.train()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        total_examples = total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_size = batch['article'].batch_size
            out = model({k: v.float() for k, v in batch.x_dict.items()}, batch.edge_index_dict)
            loss = F.binary_cross_entropy(out['article'][:batch_size],
                                          batch['article'].y[:batch_size])
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
        print(total_loss / total_examples)

        train_loss = total_loss / total_examples
        val_loss = evaluate(model, validation_loader)

        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    return best_val_loss


def load_twitter_data(results_path):
    tweets_path = os.path.join(results_path, 'tweets.csv')
    users_path = os.path.join(results_path, 'users.csv')
    articles_path = os.path.join(results_path, 'articles_df.csv')
    sources_path = os.path.join(results_path, 'sources.csv')
    hashtags_path = os.path.join(results_path, 'hashtag_df.csv')
    urls_path = os.path.join(results_path, 'url_df.csv')

    import time

    start = time.time()

    df = pd.read_csv(tweets_path, index_col='id').sample(frac=1, random_state=42)
    tweet_x, tweet_mapping = load_node_csv(df, index_col='id', encoders={
        'created_at': DateEncoder(),
        'truncated': DefaultEncoder(),
        'retweet_count': DefaultEncoder(),
        'favorite_count': DefaultEncoder(),
        'favorited': DefaultEncoder(),
        'retweeted': DefaultEncoder(),
        'is_quote_status': DefaultEncoder(),
    })
    tweet_column_max = tweet_x.max(dim=0).values
    tweet_column_max[tweet_column_max == 0] = 1
    tweet_x = tweet_x / tweet_column_max
    end = time.time()
    print("loaded tweets", end - start)

    df = pd.read_csv(users_path, index_col='id').sample(frac=1, random_state=42)
    user_x, user_mapping = load_node_csv(df, index_col='id', encoders={
        'created_at': DateEncoder(),
        'followers_count': DefaultEncoder(),
        'verified': DefaultEncoder(),
        'protected': DefaultEncoder(),
        'friends_count': DefaultEncoder(),
        'listed_count': DefaultEncoder(),
        'default_profile_image': DefaultEncoder(),
        'default_profile': DefaultEncoder(),
        'statuses_count': DefaultEncoder(),
        'favourites_count': DefaultEncoder(),
        'description': SequenceEncoder()
    })
    user_column_max = user_x.max(dim=0).values
    user_column_max[user_column_max == 0] = 1

    user_x = user_x / user_column_max
    end = time.time()
    print("loaded users", end - start)

    df = pd.read_csv(articles_path, index_col='article_dir').sample(frac=1, random_state=42)
    labels = df['label']
    df = df.drop(columns=['label'])
    article_x, article_mapping = load_node_csv(df, index_col='article_dir', encoders={
        'content_text': SequenceEncoder(),
        'title': SequenceEncoder(),
        'n_images': DefaultEncoder()
    })
    end = time.time()
    print("loaded articles", end - start)

    article_column_max = article_x.max(dim=0).values
    article_column_max[article_column_max == 0] = 1
    article_x = article_x / article_column_max

    df = pd.read_csv(sources_path, index_col='id').sample(frac=1, random_state=42)
    source_x, source_mapping = load_node_csv(df, index_col='id', encoders={
        'source': SequenceEncoder(),
    })
    end = time.time()
    print("loaded sources", end - start)

    source_column_max = source_x.max(dim=0).values
    source_column_max[source_column_max == 0] = 1
    source_x = source_x / source_column_max

    df = pd.read_csv(hashtags_path, index_col='id').sample(frac=1, random_state=42)
    hashtag_x, hashtag_mapping = load_node_csv(df, index_col='id', encoders={
        'hashtag': SequenceEncoder(),
    })
    end = time.time()
    print("loaded hashtags", end - start)

    hashtag_column_max = hashtag_x.max(dim=0).values
    hashtag_column_max[hashtag_column_max == 0] = 1
    hashtag_x = hashtag_x / hashtag_column_max

    df = pd.read_csv(urls_path, index_col='id').sample(frac=1, random_state=42)
    df['temp'] = 0
    url_x, url_mapping = load_node_csv(df, index_col='id', encoders={
        'temp': DefaultEncoder(),
    })
    end = time.time()
    print("loaded urls", end - start)

    data = HeteroData()
    data['article'].x = article_x
    data['tweet'].x = tweet_x
    data['user'].x = user_x
    data['source'].x = source_x
    data['hashtag'].x = hashtag_x
    data['url'].x = url_x

    data['article'].y = torch.tensor(labels, dtype=torch.float).unsqueeze(dim=-1)

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'tweets.csv'),
        src_index_col='id',
        src_mapping=tweet_mapping,
        dst_index_col='article_dir',
        dst_mapping=article_mapping,
    )
    end = time.time()
    print("loaded edge", end - start)

    data['tweet', 'relates', 'article'].edge_index = edge_index
    data['tweet', 'relates', 'article'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'user_tweet.csv'),
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['user', 'creates', 'tweet'].edge_index = edge_index
    data['user', 'creates', 'tweet'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'hashtag_tweet_df.csv'),
        src_index_col='hashtag_id',
        src_mapping=hashtag_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['hashtag', 'included_in', 'tweet'].edge_index = edge_index
    data['hashtag', 'included_in', 'tweet'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'source_tweet.csv'),
        src_index_col='source_id',
        src_mapping=source_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['source', 'originates', 'tweet'].edge_index = edge_index
    data['source', 'originates', 'tweet'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'tweet_reply_tweet.csv'),
        src_index_col='source_tweet_id',
        src_mapping=tweet_mapping,
        dst_index_col='reply_tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['tweet', 'replies', 'tweet'].edge_index = edge_index
    data['tweet', 'replies', 'tweet'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'url_tweet_df.csv'),
        src_index_col='url_id',
        src_mapping=url_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['url', 'is_contained', 'tweet'].edge_index = edge_index
    data['url', 'is_contained', 'tweet'].edge_label = edge_label

    edge_index, edge_label = load_edge_csv(
        os.path.join('results', 'user_mention.csv'),
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
    )

    data['user', 'is_mentioned', 'tweet'].edge_index = edge_index
    data['user', 'is_mentioned', 'tweet'].edge_label = edge_label

    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)

    return data, article_mapping


if __name__ == '__main__':
    results_path = os.path.join('results')
    data, article_mapping = load_twitter_data(results_path)
    data_idx = list(article_mapping.values())
    train_mask, validation_mask = train_test_split(data_idx, random_state=42)


    def fitness_func(instance, solution, sol_idx):
        train_loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            batch_size=len(train_mask),
            input_nodes=('article', torch.tensor(train_mask))
        )
        validation_loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            batch_size=len(validation_mask),
            input_nodes=('article', torch.tensor(validation_mask))
        )

        model, optimizer = create_model(solution)
        val_accuracy = train(model, train_loader, validation_loader, optimizer)
        print("done fitness")
        return val_accuracy


    population_size = 100
    num_generations = 20
    num_parents_mating = 10

    gene_space = [list(range(3)),  # optimizer type idx
                  list(range(5)),  # aggregation type idx
                  {'low': 1, 'high': 1000},  # learning_rate * 1000
                  {'low': 1, 'high': 10},  # heads
                  {'low': 1, 'high': 300}]  # neurons

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           gene_space=gene_space,
                           sol_per_pop=num_parents_mating * 2,
                           num_genes=len(gene_space),
                           parent_selection_type="rank",
                           crossover_type="two_points",
                           mutation_type="random",
                           mutation_percent_genes=max(10, int(1 / len(gene_space) * 100)),
                           random_mutation_min_val=1,
                           random_mutation_max_val=100,
                           mutation_by_replacement=True,
                           # random_mutation_by_replacement_max_val=100
                           )

    ga_instance.run()

    # population.best_solution()
