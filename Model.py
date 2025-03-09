import osmnx as ox
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

# TODO разобраться подробнее с библиотекой torch_geometric
# TODO выровнять длины истинных и ложных последовательностей координат
# TODO разобраться, как использовать графы при обучении модели
# TODO сделать обучающую выборку в виде файла json [DONE]
# TODO придумать, в каком виде будет храниться выборка [DONE]

"""
Модуль непосредственно самой модели нейронной сети,
основанной (предварительно) на комбинации рекуррентных нейронных сетей (RNN)
и графовых нейронных сетей (GNN).
"""


G = ox.graph_from_bbox((39.121447, 51.646002, 39.135578, 51.653782))

# Извлекаем узлы и их признаки
node_features = torch.stack([data for _, data in G.nodes(data=True)])
node_indices = {node: idx for idx, node in enumerate(G.nodes())}

# Извлекаем рёбра и их признаки
edge_indices = []
edge_features = []
for u, v, key, data in G.edges(keys=True, data=True):
    edge_indices.append([node_indices[u], node_indices[v]])
    edge_features.append(data['feature'])

edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
edge_attr = torch.stack(edge_features)

# Создаем объект Data
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

print(data)


class CoordinateCorrectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CoordinateCorrectionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Используем LSTM для обработки последовательности
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Полносвязный слой для преобразования выхода LSTM в координаты
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Инициализация скрытого состояния и состояния ячейки
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Проход через LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Применяем полносвязный слой к каждому элементу последовательности
        out = self.fc(out)

        return out


class RouteDataset(Dataset):
    def __init__(self, noisy_routes, true_routes):
        self.noisy_routes = noisy_routes  # Список последовательностей с помехами
        self.true_routes = true_routes  # Список истинных последовательностей

    def __len__(self):
        return len(self.noisy_routes)

    def __getitem__(self, idx):
        noisy = torch.tensor(self.noisy_routes[idx], dtype=torch.float32)
        true = torch.tensor(self.true_routes[idx], dtype=torch.float32)
        return noisy, true


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Объединение LSTM и GNN
class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_output_size, gnn_input_dim, gnn_hidden_dim,
                 gnn_output_dim):
        super(CombinedModel, self).__init__()
        self.lstm = CoordinateCorrectionModel(lstm_input_size, lstm_hidden_size, lstm_output_size)
        self.gnn = GNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim)
        self.fc = nn.Linear(lstm_output_size + gnn_output_dim, lstm_output_size)

    def forward(self, x, edge_index):
        lstm_out = self.lstm(x)
        gnn_out = self.gnn(x, edge_index)
        combined = torch.cat((lstm_out, gnn_out), dim=-1)
        out = self.fc(combined)
        return out


# Пример данных
noisy_routes = [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
true_routes = [[[0.0, 0.0], [0.3, 0.3], [0.6, 0.6]], [[0.7, 0.7], [1.0, 1.0], [1.3, 1.3]]]

dataset = RouteDataset(noisy_routes, true_routes)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Определение модели, функции потерь и оптимизатора
model = CombinedModel(input_size=2, hidden_size=128, output_size=2, num_layers=2)
criterion = nn.MSELoss()  # Функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор

num_epochs = 10  # Количество эпох

for epoch in range(num_epochs):
    for batch_noisy, batch_true in dataloader:
        # Forward pass
        outputs = model(batch_noisy)
        loss = criterion(outputs, batch_true)

        # Backward pass и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
