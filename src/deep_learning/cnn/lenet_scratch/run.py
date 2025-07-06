from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import override

from src.utils.configuration import BaseConfig
from src.utils.trainer import BaseModel, ModelOutput, Trainer


@dataclass
class TrainConfig(BaseConfig):
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    output_dir: str


class LeNet(BaseModel):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 输入通道1，输出6个特征图
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    @override
    def forward(self, batch) -> ModelOutput:
        x, label = batch
        x = self.features(x)
        x = self.classifier(x)
        return ModelOutput(pred=x, label=label)


def accuracy(pred, label):
    return (pred.argmax(axis=-1) == label).mean().item()


if __name__ == "__main__":
    config = TrainConfig.load_file(Path(__file__).parent / "config.yaml")
    # 数据预处理和加载
    transform = T.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root=Path("dataset"), train=True, transform=transform, download=True)
    train_size, val_size, test_size = 35000, 20000, 5000
    train_subset, val_subset, test_subset = random_split(train_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_subset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.train_batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=config.eval_batch_size, shuffle=False)
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    metrics = {"acc": accuracy}
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        scheduler=scheduler,
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        metrics=metrics,
    )
    trainer.save_checkpoint(Path(config.output_dir))
    result = trainer.evaluate(test_loader, metrics=metrics)
    print(f"Test accuracy: {result['acc']:.4f}")
