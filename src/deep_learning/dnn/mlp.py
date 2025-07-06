from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from src.utils.configuration import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    output_dir: str

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    config = TrainConfig.load_file(Path(__file__).parent / "config.yaml")
    # 数据预处理和加载
    transform = T.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root=Path("dataset"), train=True, transform=transform, download=True)
    train_size, val_size, test_size = 35000, 20000, 5000
    train_subset, val_subset, test_subset = random_split(train_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_subset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.eval_batch_size, shuffle=False)
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环（仅训练一个epoch作为示例）
    for epoch in range(config.num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1:2d}") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1:2d}") as pbar:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    pbar.update(1)
                    pbar.set_postfix(val_loss=f"{loss.item():.4f}")
                accuracy = correct / len(val_loader.dataset)
                pbar.set_postfix(val_loss=loss.item(), val_accuracy=accuracy)

    test_loader = DataLoader(test_subset, batch_size=config.eval_batch_size, shuffle=False)
    model.eval()
    correct = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluation: ") as pbar:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                pbar.update(1)
            accuracy = correct / len(test_loader.dataset)
            pbar.set_postfix(val_accuracy=accuracy)
    print(f"Test accuracy: {accuracy:.4f}")
