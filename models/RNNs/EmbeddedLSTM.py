import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, coords, coords_lengths):
        embedded = self.embedding(coords)
        packed_embedded = pack_padded_sequence(embedded, coords_lengths.cpu(),
                                               batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output[:, -1, :])
