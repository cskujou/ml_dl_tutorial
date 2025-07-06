import torch
from pathlib import Path

from src.utils.trainer import Evaulator
from src.utils.dataset import ImageFolderDataset, get_transform

from .train import PlantDocClassifier, TrainConfig, accuracy


if __name__ == "__main__":
    config = TrainConfig.load_file(Path(__file__).parent / "config.yaml", override_with_args=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolderDataset(Path("dataset") / "PlantDoc-Dataset", splits="test")
    model = PlantDocClassifier(num_classes=len(dataset.id2label))
    mcfg = model.model.pretrained_cfg
    input_size, mean, std = mcfg["input_size"][1:], mcfg["mean"], mcfg["std"]
    test_transform = get_transform(input_size, mean, std)
    test_loader = dataset.get_loader(
        "test",
        transform=test_transform,
        persistent=True,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
    )

    evaulator = Evaulator(model=model, device=device)
    evaulator.load_checkpoint(Path(config.output_dir))

    metrics = {"acc": accuracy}
    results = evaulator.evaluate(test_loader, metrics=metrics)
    print(f"Test accuracy: {results['acc']:.4f}")
