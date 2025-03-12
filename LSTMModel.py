# TODO разобраться подробнее с библиотекой torch_geometric
#  -> конвертировать networkx.MultiDiGraph в torch_geometric.data.Data
# TODO выровнять длины истинных и ложных последовательностей координат [DONE]
# TODO разобраться, как использовать графы при обучении модели
# TODO сделать обучающую выборку в виде файла json [DONE]
# TODO придумать, в каком виде будет храниться выборка [DONE]
# TODO обучить и сравнить следующие модели: LSTM (+ добавить эмбеддинг длин последовательностей),
#  GNN, seq2seq, ансамбль LSTM и GNN, ансамбль seq2seq и GNN
# TODO дополнительно посмотреть про GeoGNN

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict

with (open('test.json', 'r', encoding='utf-8')) as f:
    data = json.load(f)

X, y = data['X'], data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=42)


# Определяем модель LSTM
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
        # Проходим через LSTM
        out, _ = self.lstm(x)
        # Проходим через линейный слой
        out = self.fc(out[:, -1, :])
        return out


# Параметры модели
input_size = 2  # Широта и долгота
hidden_size = 64
output_size = 2
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создаём модель
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32

loss_fn = nn.MSELoss(reduction='mean')

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

# Обучение модели
num_epochs = 500
train_hist = []
test_hist = []

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_hist.append(average_loss)

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

    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')
