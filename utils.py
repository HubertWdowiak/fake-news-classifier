import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DefaultEncoder:
    def __call__(self, x):
        return torch.tensor(x.values).unsqueeze(1).to(device)


class DateEncoder:
    def __call__(self, x):
        df = pd.DataFrame()
        df['year'] = x.apply(lambda a: pd.to_datetime(a).year)
        df['month'] = x.apply(lambda a: pd.to_datetime(a).month)
        df['day'] = x.apply(lambda a: pd.to_datetime(a).day)
        df['hour'] = x.apply(lambda a: pd.to_datetime(a).hour)
        df['minute'] = x.apply(lambda a: pd.to_datetime(a).minute)
        df['second'] = x.apply(lambda a: pd.to_datetime(a).second)
        return torch.tensor(df.values).to(device)


class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        df.replace(np.nan, '', inplace=True)
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.to(self.device)


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



def load_node_csv(df, encoders=None, **kwargs):
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1).to(device)

    return x, mapping

