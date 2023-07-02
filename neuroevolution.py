import torch
from torch.optim import Adam
from torch.optim import SGD, RMSprop
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, Linear, to_hetero
import pygad
from torch.functional import F


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=heads)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=heads)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return torch.sigmoid(x)


def get_optimizer(idx):
    optimizers = [Adam, SGD, RMSprop]
    return optimizers[idx]


def get_aggregation(idx):
    aggrs = ['sum', 'mean', 'max', 'min', 'mul']
    return aggrs[idx]


def create_model(solution):
    optimizer_type = get_optimizer(solution[0])
    aggregation = get_aggregation(solution[1])
    learning_rate = solution[2]
    heads = solution[3]
    neurons = solution[4]

    model = GAT(hidden_channels=neurons, out_channels=1, heads=heads)
    model = to_hetero(model, data.metadata(), aggr=aggregation)
    optimizer = optimizer_type(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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


def train(model, train_loader, validation_loader, optimizer, epochs: int = 1000):
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


if __name__ == '__main__':
    data = ...

    X_train = data[0]
    X_test = data[1]
    train_mask = ...
    validation_mask = ...

    def fitness_func(solution, sol_idx):
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
            input_nodes=('article', torch.tensor(train_mask))
        )

        model, optimizer = create_model(solution)
        val_accuracy = train(model, train_loader, validation_loader, optimizer)
        return val_accuracy

    population_size = 50
    num_generations = 10
    num_parents_mating = 10

    gene_space = [{'low': 0, 'high': 3},  # optimizer type idx
                  {'low': 0, 'high': 5},  # aggregation type idx
                  {'low': 1, 'high': 1000},  # learning_rate * 1000
                  {'low': 1, 'high': 10},  # heads
                  {'low': 1, 'high': 300}]  # neurons

    population = pygad.GA(initial_population=None,
                          num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          fitness_func=fitness_func,
                          gene_space=gene_space,
                          parent_selection_type="rank",
                          crossover_type="two_points",
                          mutation_type="random",
                          mutation_percent_genes=10,
                          random_mutation_min_val=1,
                          random_mutation_max_val=100,
                          mutation_by_replacement=True,
                          # random_mutation_by_replacement_max_val=100
                          )

    population.run()

    # population.best_solution()
