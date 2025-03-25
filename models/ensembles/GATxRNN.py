import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import geopandas as gpd
import numpy as np
from torch_geometric.data import Data
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class RouteCorrector(nn.Module):
    """"
    Класс модели, которая использует графовую нейронную сеть с механизмом внимания
    и рекуррентную нейронную сеть на основе seq2seq
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Кодировщик координат (преобразует (lat, lon) в эмбеддинг)
        self.coord_encoder = nn.Linear(2, hidden_dim)

        # Кодировщик графа (GNN)
        self.gnn = GATConv(hidden_dim, hidden_dim)

        # Механизм внимания между точками и графом
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Seq2seq, предсказывает следующий узел
        self.decoder = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, graph_data, route_coords):
        route_emb = self.coord_encoder(route_coords)  # [seq_len, hidden_dim]
        node_emb = self.gnn(graph_data.x, graph_data.edge_index)  # [num_nodes, hidden_dim]

        # 3. Сопоставляем точки маршрута с узлами графа через внимание
        corrected_emb, _ = self.attention(
            route_emb, node_emb, node_emb
        )

        # 4. Декодируем последовательность узлов
        output, _ = self.decoder(corrected_emb)
        return output  # [seq_len, hidden_dim]


# Загрузка графа дорог Воронежа
dirpath = os.path.dirname(__file__)
datapath = os.path.join(dirpath, "..\\..\\training data")

# Атрибуты вершины: координаты, id, кол-вол улиц
nodes = gpd.read_file(os.path.join(datapath, "nodes.csv"), encoding="utf8")
nodes = nodes.iloc[:, :4]
nodes['osmid'], nodes['street_count'] = (np.array(nodes['osmid'], dtype=np.int64),
                                         np.array(nodes['street_count'], dtype=np.int64))
nodes['y'], nodes['x'] = (np.array(nodes['y'], dtype=np.float64),
                          np.array(nodes['x'], dtype=np.float64))

edges = gpd.read_file(os.path.join(datapath, "edges.csv"), encoding="utf8")
edge_index = edges[['u', 'v']]
# Атрибуты ребра: id, кол-во полос, односторонность, реверсивность, длина
edge_attr = edges[['osmid', 'lanes', 'oneway', 'reversed', 'length']]

# Загрузка маршрутов
routes_path = os.path.join(dirpath, "..\\..\\training data\\routes.json")
with (open(routes_path, 'r', encoding='utf-8')) as f:
    data = json.load(f)

X, y = data['X'], data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=42)
sc = MinMaxScaler(feature_range=(-1, 1))
nodes_t = torch.tensor(nodes.values)
edge_index_t = torch.tensor(edge_index.values)
edge_attr_t = torch.tensor(edge_attr.values)
