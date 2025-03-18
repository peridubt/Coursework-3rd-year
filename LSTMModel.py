import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


if __name__ == "__main__":
    with (open('test.json', 'r', encoding='utf-8')) as f:
        data = json.load(f)

    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=42)
    sc = MinMaxScaler(feature_range=(-1, 1))  # для нормализации данных от -1 до 1

    # Параметры модели
    input_size = 2  # Выходная размерность последовательности: Широта и долгота
    hidden_size = 128
    output_size = 2
    num_layers = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # создаём модель
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)

    # функция потерь и оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    batch_size = 32
    loss_fn = nn.L1Loss()

    X_train = [torch.tensor(sc.fit_transform(seq), dtype=torch.float32) for seq in X_train]
    y_train = [torch.tensor(sc.fit_transform(seq), dtype=torch.float32) for seq in y_train]
    X_train_pad = pad_sequence(X_train, batch_first=True)
    y_train_pad = pad_sequence(y_train, batch_first=True)

    X_test = [torch.tensor(sc.fit_transform(seq), dtype=torch.float32) for seq in X_test]
    y_test = [torch.tensor(sc.fit_transform(seq), dtype=torch.float32) for seq in y_test]
    X_test_pad = pad_sequence(X_test, batch_first=True)
    y_test_pad = pad_sequence(y_test, batch_first=True)

    train_dataset = TensorDataset(X_train_pad, y_train_pad)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_pad, y_test_pad)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # обучение модели
    num_epochs = 10  # Количество эпох при обучении
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

        print(
            f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')
    torch.save(model, 'lstm_model.pth')
