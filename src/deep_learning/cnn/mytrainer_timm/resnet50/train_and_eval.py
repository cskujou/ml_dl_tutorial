from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from pathlib import Path

from src.utils.configuration import BaseConfig
from src.utils.trainer import BaseModel, ModelOutput, Trainer
from src.utils.dataset import ImageFolderDataset, get_transform
import timm


@dataclass
class TrainConfig(BaseConfig):
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    output_dir: str
    num_workers: int = 0
    transform_on_the_fly: bool = False


class PlantDocClassifier(BaseModel):
    def __init__(self, num_classes):
        super(PlantDocClassifier, self).__init__()
        model_name = "resnet50.a1_in1k"
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            pretrained_cfg_overlay=dict(file=f"model/timm/{model_name}/model.safetensors"),
        )

    def forward(self, batch) -> ModelOutput:
        x, label = batch
        x = self.model(x)
        return ModelOutput(pred=x, label=label)


def accuracy(pred, label):
    return (pred.argmax(axis=-1) == label).mean().item()


if __name__ == "__main__":
    config = TrainConfig.load_file(Path(__file__).parent / "config.yaml", override_with_args=True)

    # 数据预处理和加载
    dataset = ImageFolderDataset(Path("dataset") / "PlantDoc-Dataset", splits="train")
    dataset.split(children=["train", "val"], ratios=[0.8, 0.2], parent="train", shuffle=True)
    model = PlantDocClassifier(num_classes=dataset.num_classes())

    mcfg = model.model.pretrained_cfg
    input_size, mean, std = mcfg["input_size"][1:], mcfg["mean"], mcfg["std"]

    augs = [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    ]
    train_transform = get_transform(input_size=input_size, mean=mean, std=std, random_resized=True, augs=augs)
    val_transform = get_transform(input_size=input_size, mean=mean, std=std)

    train_loader = dataset.get_loader(
        "train",
        batch_size=config.train_batch_size,
        transform=train_transform,
        transform_on_the_fly=config.transform_on_the_fly,
        shuffle=True,
        persistent=True,
        num_workers=config.num_workers,
    )
    val_loader = dataset.get_loader(
        "val",
        batch_size=config.eval_batch_size,
        transform=val_transform,
        shuffle=False,
        persistent=True,
        num_workers=config.num_workers,
    )
    test_loader = dataset.get_loader(
        "test",
        batch_size=config.eval_batch_size,
        transform=val_transform,
        shuffle=False,
        persistent=True,
        num_workers=config.num_workers,
    )
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)
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
    results = trainer.evaluate(data_loader=test_loader, criterion=criterion, metrics=metrics)
    print(f"Test accuracy: {results['acc']:.4f}")
