from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import gzip
import torch
import random
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as T


@dataclass
class ImageSample:
    path: Path
    label: int | None = None


class ImageSplitDataset(Dataset):
    def __init__(
        self,
        samples: list[ImageSample],
        root_dir: Path,
        split: str,
        transform: Any,
        transform_on_the_fly: bool = False,
        persistent: bool = False,
        cache_transformed: bool = False,
    ):
        self.samples: list[ImageSample] = samples
        self.root_dir: str = root_dir
        self.split: str = split
        self.transform = transform
        self.transform_on_the_fly = transform_on_the_fly
        self.persistent: bool = persistent
        self.cache: dict[int, torch.Tensor] = {}
        self.cache_transformed = cache_transformed
        if self.cache_transformed and self.transform_on_the_fly:
            print("WARNING: `cache_preprocessed` will be ignored when `transform_on_the_fly` is True")
        if self.persistent:
            with tqdm(total=len(self.samples), desc="Loading images") as pbar:
                for i, sample in enumerate(self.samples):
                    self.cache[i] = self._load_image(sample.path)
                    pbar.update()

    def _load_image(self, path: Path) -> torch.Tensor:
        # 如果缓存文件存在，则从缓存文件中加载图像
        cache_file = self.root_dir / ".cache" / self.split / f"{path.parent.stem}-{path.stem}.gz"
        if self.cache_transformed and cache_file.exists():
            with gzip.open(cache_file, "rb") as f:
                image = torch.load(f)
        # 否则，从原始文件中加载图像，并进行转换
        else:
            image = decode_image(path, mode="RGB") / 255  # [C, H, W]
            if self.transform is not None and not self.transform_on_the_fly:
                image = self.transform(image)
                if self.cache_transformed:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(cache_file, "wb") as f:
                        torch.save(image, f)
        return image

    def preload(self, start: int | None = None, end: int | None = None):
        start = 0 if start is None else start
        end = len(self.samples) if end is None else end
        for i in range(start, end):
            self.cache[i] = self._load_image(self.samples[i].path)

    def release(self, start: int | None = None, end: int | None = None):
        start = 0 if start is None else start
        end = len(self.samples) if end is None else end
        for i in range(start, end):
            if i in self.cache:
                del self.cache[i]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        sample = self.samples[index]
        if index in self.cache:
            image = self.cache[index]
        else:
            image = self._load_image(sample.path)

        if self.transform is not None and self.transform_on_the_fly:
            image = self.transform(image)
        return image, sample.label


class ImageFolderDataset:
    def __init__(self, root_dir: Path, splits: str | list[str] | None = None):
        self.root_dir = root_dir
        self.split_dict = {}
        self._load_splits(splits)

    def _load_splits(self, splits):
        init_target_labels = False
        if splits is None:
            splits = ["default"]
        if isinstance(splits, str):
            splits = [splits]
        for split in splits:
            if split == "default":
                split_path = self.root_dir
            else:
                split_path = self.root_dir / split
            samples, id2label, label2id = self._load_split(split_path)
            if not init_target_labels:
                self.id2label = id2label
                self.label2id = label2id
                init_target_labels = True
            else:
                assert id2label == self.id2label, "Mismatched labels between splits"
            self.split_dict[split] = samples

    def num_classes(self):
        return len(self.id2label)

    def _load_split(self, split_path):
        samples = []
        id2label, label2id = [], {}
        folders = [x for x in split_path.iterdir() if x.is_dir()]
        if len(folders) > 0:
            for label, folder_dir in enumerate(folders):
                id2label.append(folder_dir.name)
                label2id[folder_dir.name] = label
                samples.extend(self._load_folder(folder_dir, label))
        else:
            # 可能是个无标签的测试集
            samples = self._load_folder(split_path, label=None)

        return samples, id2label, label2id

    def _load_folder(self, folder_dir, label=None):
        return [ImageSample(image_path, label) for image_path in folder_dir.iterdir() if image_path.is_file()]

    def add_split_from_folder(self, split_path: Path | str, split: str):
        samples, id2label, _ = self._load_split(split_path)
        self.split_dict[split] = samples
        assert id2label == self.id2label, "Mismatched labels between splits"

    def split(
        self,
        children: list[str] | tuple[str],
        ratios: list[float] | tuple[float],
        parent: str = "default",
        shuffle: bool = False,
        seed: int = 2025,
    ):
        if len(children) != len(ratios):
            raise ValueError("Mismatched number of childs and ratios")
        if sum_ratios := sum(ratios) != 1:
            ratios = [r / sum_ratios for r in ratios]
        if parent == "default" and parent not in self.split_dict:
            if len(self.split_dict) == 1:
                parent = list(self.split_dict.keys())[0]
            else:
                raise ValueError("You must specify a parent split")
        elif parent not in self.split_dict:
            raise ValueError(f"Parent split {parent} not found")
        parent_samples = self.split_dict[parent]
        num_samples = len(parent_samples)
        if shuffle:
            random.shuffle(parent_samples)
        split_points = [0] + [int(sum(ratios[: i + 1]) * num_samples) for i in range(len(ratios) - 1)] + [num_samples]
        for i, (start, end) in enumerate(pairwise(split_points)):
            self.split_dict[children[i]] = parent_samples[start:end]
        if parent not in children:
            del self.split_dict[parent]

    def get_loader(
        self,
        split: str,
        transform: Any = None,
        transform_on_the_fly: bool = False,
        persistent: bool = True,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **dataloader_kwargs,
    ):
        dataset = ImageSplitDataset(
            self.split_dict[split],
            root_dir=self.root_dir,
            split=split,
            transform=transform,
            transform_on_the_fly=transform_on_the_fly,
            persistent=persistent,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **dataloader_kwargs,
        )


def get_transform(
    input_size: tuple[int] | list[int],
    mean: tuple[int] | list[int] | None = None,
    std: tuple[int] | list[int] | None = None,
    random_resized=False,
    augs: list[callable] | None = None,
):
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    augs = [] if augs is None else augs
    crop = T.RandomResizedCrop if not random_resized else T.Resize
    return T.Compose(
        [
            *augs,
            crop(input_size, antialias=True),
            T.Normalize(mean=mean, std=std),
        ]
    )
