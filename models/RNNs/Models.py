"""
Модуль, где описаны классы разных моделей
"""
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # проходим через LSTM
        out, _ = self.lstm(x)
        # проходим через линейный слой
        out = self.fc(out)
        return out




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
