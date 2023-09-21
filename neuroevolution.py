import os
import torch_geometric.transforms as T

import pandas as pd
import pygad
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim import SGD, RMSprop
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch.nn import BCEWithLogitsLoss, Dropout
from torch_geometric.nn import GATConv, Linear, to_hetero
import wandb

from utils import load_node_csv, DateEncoder, DefaultEncoder, SequenceEncoder, load_edge_csv, get_min_and_max, \
    normalize_batch, get_class_weight_ratio

wandb.login(key='')
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=False, dropout=dropout)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=heads, concat=False, dropout=dropout)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.sigmoid()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


def get_optimizer(idx):
    optimizers = [Adam, SGD, RMSprop]
    return optimizers[int(idx)]


def get_aggregation(idx):
    aggrs = ['sum', 'mean', 'max', 'min', 'mul']
    return aggrs[int(idx)]


def create_model(solution):
    optimizer_type = get_optimizer(solution[0])
    aggregation = get_aggregation(solution[1])
    learning_rate = int(solution[2]) / 1000
    heads = int(solution[3])
    neurons = int(solution[4])
    dropout = solution[5] / 100

    model = GAT(hidden_channels=neurons, out_channels=1, heads=heads, dropout=dropout)
    model = to_hetero(model, data.metadata(), aggr=aggregation).to(device)
    optimizer = optimizer_type(lr=learning_rate, params=model.parameters())

    return model, optimizer


def evaluate(model, data_loader, class_weight, min_vals, max_vals):
    model.eval()
    total_examples = total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            normalized_batch = normalize_batch(batch, min_vals, max_vals)
            batch_size = normalized_batch['article'].batch_size
            out = model({k: v.float() for k, v in normalized_batch.x_dict.items()}, normalized_batch.edge_index_dict)
            loss_function = BCEWithLogitsLoss(pos_weight=class_weight)
            loss = loss_function(out['article'][:batch_size],
                                 normalized_batch['article'].y[:batch_size])
            total_examples += batch_size
            total_loss += float(loss) * batch_size

    return total_loss / total_examples


def train(model: GAT, train_loader: NeighborLoader, validation_loader: NeighborLoader, optimizer: torch.optim.Optimizer,
          epochs: int, min_vals: torch.Tensor, max_vals: torch.Tensor, class_weight: torch.Tensor, patience: int = 10):
    model.train()
    best_val_loss = float('inf')

    wandb.init(
        project="fake-news",
        config={
            "lr": optimizer.param_groups[0]['lr'],
            'heads': model.conv2['tweet__relates__article'].heads,
            'optimizer': optimizer.__class__.__name__,
            'neurons': model.conv1['tweet__relates__article'].out_channels,
            'aggregation': model.conv1['tweet__relates__article'].aggr,
            "dropout": model.conv1['tweet__relates__article'].dropout
        })

    for epoch in range(epochs):
        total_examples = total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            normalized_batch = normalize_batch(batch, min_vals, max_vals)
            optimizer.zero_grad()
            batch_size = normalized_batch['article'].batch_size
            out = model({k: v.float() for k, v in normalized_batch.x_dict.items()}, normalized_batch.edge_index_dict)
            loss_function = BCEWithLogitsLoss(pos_weight=class_weight)
            loss = loss_function(out['article'][:batch_size],
                                 normalized_batch['article'].y[:batch_size])
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
        print(total_loss / total_examples)

        train_loss = total_loss / total_examples
        val_loss = evaluate(model, validation_loader, class_weight, min_vals, max_vals)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0  # Reset patience counter when we find a better model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping. No improvement for {patience} epochs.")
                break  # Stop training

        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    wandb.finish()

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
    },
                                           device=device)
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
        'description': SequenceEncoder(device=device)
    },
                                         device=device)
    user_column_max = user_x.max(dim=0).values
    user_column_max[user_column_max == 0] = 1

    user_x = user_x / user_column_max
    end = time.time()
    print("loaded users", end - start)

    df = pd.read_csv(articles_path, index_col='article_dir').sample(frac=1, random_state=42)
    labels = df['label']
    df = df.drop(columns=['label'])
    article_x, article_mapping = load_node_csv(df, index_col='article_dir', encoders={
        'content_text': SequenceEncoder(device=device),
        'title': SequenceEncoder(device=device),
        'n_images': DefaultEncoder()
    },
                                               device=device)
    end = time.time()
    print("loaded articles", end - start)

    article_column_max = article_x.max(dim=0).values
    article_column_max[article_column_max == 0] = 1
    article_x = article_x / article_column_max

    df = pd.read_csv(sources_path, index_col='id').sample(frac=1, random_state=42)
    source_x, source_mapping = load_node_csv(df, index_col='id', encoders={
        'source': SequenceEncoder(),
    },
                                             device=device)
    end = time.time()
    print("loaded sources", end - start)

    source_column_max = source_x.max(dim=0).values
    source_column_max[source_column_max == 0] = 1
    source_x = source_x / source_column_max

    df = pd.read_csv(hashtags_path, index_col='id').sample(frac=1, random_state=42)
    hashtag_x, hashtag_mapping = load_node_csv(df, index_col='id', encoders={
        'hashtag': SequenceEncoder(),
    },
                                               device=device)
    end = time.time()
    print("loaded hashtags", end - start)

    hashtag_column_max = hashtag_x.max(dim=0).values
    hashtag_column_max[hashtag_column_max == 0] = 1
    hashtag_x = hashtag_x / hashtag_column_max

    df = pd.read_csv(urls_path, index_col='id').sample(frac=1, random_state=42)
    df['temp'] = 0
    url_x, url_mapping = load_node_csv(df, index_col='id', encoders={
        'temp': DefaultEncoder(),
    },
                                       device=device)
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
    # data, article_mapping = load_twitter_data(results_path)

    # torch.save(data, 'graph.pt')
    # torch.save(article_mapping, 'mapping.pt')

    data, article_mapping = torch.load('graph.pt', map_location=torch.device('cpu')), torch.load('mapping.pt',
                                                                                                 map_location=torch.device(
                                                                                                     'cpu'))

    data = data.edge_type_subgraph(
        [('tweet', 'relates', 'article'), ('user', 'creates', 'tweet'), ('hashtag', 'included_in', 'tweet'),
         ('source', 'originates', 'tweet'), ('tweet', 'replies', 'tweet'), ('url', 'is_contained', 'tweet'),
         ('article', 'rev_relates', 'tweet'), ('tweet', 'rev_creates', 'user'), ('tweet', 'rev_included_in', 'hashtag'),
         ('tweet', 'rev_originates', 'source'), ('tweet', 'rev_is_contained', 'url'), ('tweet', 'user')])

    data_idx = list(article_mapping.values())
    train_mask, validation_mask = train_test_split(data_idx, random_state=42)

    min_vals, max_vals = get_min_and_max(data, train_mask)
    class_weights = get_class_weight_ratio(data, train_mask)


    def fitness_func(instance, solution, sol_idx):
        train_loader = HGTLoader(
            data,
            num_samples={key: [512] * 4 for key in data.node_types},
            batch_size=len(train_mask) // 10,
            input_nodes=('article', torch.tensor(train_mask)),
        )
        validation_loader = HGTLoader(
            data,
            num_samples={key: [512] * 4 for key in data.node_types},
            batch_size=len(validation_mask) // 10,
            input_nodes=('article', torch.tensor(validation_mask)),
        )

        model, optimizer = create_model(solution)
        val_accuracy = train(model, train_loader, validation_loader, optimizer, 100, min_vals, max_vals, class_weights)
        print("done fitness")
        torch.cuda.empty_cache()
        return val_accuracy


    population_size = 100
    num_generations = 20
    num_parents_mating = 10

    gene_space = [list(range(3)),  # optimizer type idx
                  list(range(5)),  # aggregation type idx
                  {'low': 1, 'high': 1000},  # learning_rate * 1000
                  {'low': 1, 'high': 10},  # heads
                  {'low': 1, 'high': 300}, # neurons
                  {'low': 1, 'high': 100}]  # dropout

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
