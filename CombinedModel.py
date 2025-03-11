"""
Модуль непосредственно самой модели нейронной сети,
основанной (предварительно) на комбинации рекуррентных нейронных сетей (RNN)
и графовых нейронных сетей (GNN).
"""

import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler

import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict

# Нормализация данных
def normalize(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])


# Границы координат
lat_bounds = (39.121447, 39.135578)
lon_bounds = (51.646002, 51.653782)

# Нормализуем данные
sequences_normalized = [normalize(seq, (np.array([lat_bounds[0], lon_bounds[0]]),
                                        np.array([lat_bounds[1], lon_bounds[1]]))) for seq in sequences]

# Получаем длины последовательностей
lengths = [len(seq) for seq in sequences_normalized]

# Сортируем последовательности по убыванию длины
sorted_indices = np.argsort(lengths)[::-1]
sequences_sorted = [sequences_normalized[i] for i in sorted_indices]
lengths_sorted = [lengths[i] for i in sorted_indices]

# Преобразуем данные в тензоры PyTorch
data_tensor = torch.tensor(np.concatenate(sequences_sorted), dtype=torch.float32)

# Создаём искусственные целевые данные (для примера)
target_tensor = data_tensor.clone()

# Определяем модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Проходим через LSTM
        output, _ = self.lstm(x)
        # Проходим через линейный слой
        out = self.fc(output)
        return out

