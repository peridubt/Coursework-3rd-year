import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


# 1. Подготовка данных
class RouteDataset(Dataset):
    def __init__(self, distorted_dir, clean_dir, transform=None):
        self.distorted_dir = distorted_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.distorted_images = sorted(os.listdir(distorted_dir))
        self.clean_images = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.distorted_images)

    def __getitem__(self, idx):
        distorted_path = os.path.join(self.distorted_dir, self.distorted_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        distorted_image = Image.open(distorted_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        if self.transform:
            distorted_image = self.transform(distorted_image)
            clean_image = self.transform(clean_image)

        return distorted_image, clean_image


# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация к [-1, 1]
])

# Создание DataLoader
dataset = RouteDataset(distorted_dir="path/to/distorted", clean_dir="path/to/clean", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# 2. Архитектура модели
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Выход в диапазоне [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# 3. Функция потерь и оптимизатор
criterion = nn.L1Loss()  # или nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    for i, (distorted, clean) in enumerate(dataloader):
        distorted = distorted.to(device)
        clean = clean.to(device)

        # Forward pass
        outputs = model(distorted)
        loss = criterion(outputs, clean)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# 5. Генерация изображений
model.eval()
with torch.no_grad():
    # Загрузите искажённое изображение
    distorted_image = Image.open("path/to/distorted_image.png").convert("RGB")
    distorted_image = transform(distorted_image).unsqueeze(0).to(device)

    # Генерация исправленного изображения
    generated_image = model(distorted_image)
    generated_image = generated_image.squeeze().cpu().numpy()

    # Сохранение или визуализация generated_image
    generated_image = (generated_image * 0.5 + 0.5) * 255  # Денормализация к [0, 255]
    generated_image = generated_image.transpose(1, 2, 0).astype("uint8")
    Image.fromarray(generated_image).save("path/to/generated_image.png")
