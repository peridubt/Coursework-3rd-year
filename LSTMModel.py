# TODO разобраться подробнее с библиотекой torch_geometric
#  -> конвертировать networkx.MultiDiGraph в torch_geometric.data.Data
# TODO выровнять длины истинных и ложных последовательностей координат [DONE]
# TODO разобраться, как использовать графы при обучении модели
# TODO сделать обучающую выборку в виде файла json [DONE]
# TODO придумать, в каком виде будет храниться выборка [DONE]
# TODO обучить и сравнить следующие модели: LSTM (+ добавить эмбеддинг длин последовательностей),
#  GNN, seq2seq, ансамбль LSTM и GNN, ансамбль seq2seq и GNN, (маловероятно) CNN/GAN
# TODO дополнительно посмотреть про GeoGNN

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ImageGenerator import ImageGenerator
from RouteGenerator import RouteGenerator
from TMSRequest import TMSRequest
from collections import defaultdict

with (open('test.json', 'r', encoding='utf-8')) as f:
    data = json.load(f)

X, y = data['X'], data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=42)


# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, coords, coords_lengths):
#         embedded = self.embedding(coords)
#         packed_embedded = pack_padded_sequence(embedded, coords_lengths.cpu(),
#                                                batch_first=True, enforce_sorted=False)
#         packed_output, _ = self.lstm(packed_embedded)
#         output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
#         return self.fc(output[:, -1, :])
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # проходим через LSTM
        out, _ = self.lstm(x)
        # проходим через линейный слой
        out = self.fc(out)
        return out


def test(test_model: nn.Module, save_folder: str) -> None:
    place_bbox = [39.16064, 51.72495, 39.18008, 51.71307]
    generator = ImageGenerator(place_bbox=place_bbox)
    G, main_route = generator.graph, generator.generate_main_route()
    G_false, false_route = generator.get_one_false_route(main_route)

    geo_dict = generator.transform_bbox(place_bbox)
    tms = TMSRequest()
    tms.get(geo_dict, G, main_route, save_folder, "target", "png")
    tms.get(geo_dict, G_false, false_route, save_folder, "input", "png")

    false_coords = [[G_false.nodes[n]["x"], G_false.nodes[n]["y"]] for n in false_route]
    test_model.eval()
    with torch.no_grad():
        predict = model(torch.tensor(false_coords))
    predict = predict.detach().numpy()
    print(predict)
    tms.draw_route_on_map(predict, 15, save_folder, "predict", "png")


# Параметры модели
input_size = 2  # Выходная размерность последовательности: Широта и долгота
hidden_size = 64
output_size = 2
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# создаём модель
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# функция потерь и оптимизатор
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
loss_fn = nn.MSELoss()

X_train = [torch.tensor(seq) for seq in X_train]
y_train = [torch.tensor(seq) for seq in y_train]
X_train_pad = pad_sequence(X_train, batch_first=True)
y_train_pad = pad_sequence(y_train, batch_first=True)
# X_train_lens = torch.tensor([len(seq) for seq in X_train])
# y_train_lens = torch.tensor([len(seq) for seq in y_train])

X_test = [torch.tensor(seq) for seq in X_test]
y_test = [torch.tensor(seq) for seq in y_test]
X_test_pad = pad_sequence(X_test, batch_first=True)
y_test_pad = pad_sequence(y_test, batch_first=True)
# X_test_lens = torch.tensor([len(seq) for seq in X_test])
# y_test_lens = torch.tensor([len(seq) for seq in y_test])

train_dataset = TensorDataset(X_train_pad, y_train_pad)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_pad, y_test_pad)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# обучение модели
num_epochs = 500  # Количество эпох при обучении
train_hist = []
test_hist = []

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    for batch_X, batch_y in train_loader:  # выборка разделяется на части (батчи)
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)  # для каждого батча считается функция потерь

        # обратное распространение ошибки
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_hist.append(average_loss)

    # расчёты для тестовых бачтей
    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0

        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            predictions_test = model(batch_X_test)
            test_loss = loss_fn(predictions_test, batch_y_test)

            total_test_loss += test_loss.item()

        average_test_loss = total_test_loss / len(test_loader)
        test_hist.append(average_test_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

os.makedirs("images", exist_ok=True)
test(model, "images")
