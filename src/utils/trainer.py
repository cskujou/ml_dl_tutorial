import os
from pathlib import Path
from typing import override
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import torch

from src.utils.configuration import BaseConfig


@dataclass
class ModelOutput:
    pred: torch.Tensor = None
    label: torch.Tensor = None


class BaseModel(torch.nn.Module):
    def forward(self, batch) -> ModelOutput:
        raise NotImplementedError


class Evaulator:
    def __init__(self, model: BaseModel, device: str | torch.device = "cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def _move_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = [x.to(self.device) for x in batch]
        elif isinstance(batch, dict):
            batch = {key: value.to(self.device) for key, value in batch.items()}
        else:
            batch = batch.to(self.device)
        return batch

    @torch.no_grad()
    def _evaluate(self, data_loader, criterion, metrics, desc=None):
        self.model.eval()
        metrics = {} if metrics is None else metrics
        all_targets, all_preds = [], []
        with tqdm(total=len(data_loader), desc=desc) as pbar:
            for batch in data_loader:
                output = self.model(self._move_batch(batch))
                all_preds.append(output.pred.cpu().numpy())
                all_targets.append(output.label.cpu().numpy())
                if criterion is not None:
                    loss = criterion(output.pred, output.label)
                    pbar.set_postfix(eval_loss=loss.item())
                pbar.update()
            all_preds = np.concat(all_preds)
            eval_result = {}
            if len(all_targets) > 0:
                all_targets = np.concat(all_targets)
                eval_result = {name: metric(all_preds, all_targets) for name, metric in metrics.items()}
            eval_result |= {"eval_loss": loss.item()} if criterion is not None else {}
            pbar.set_postfix(eval_result)
        return all_preds, eval_result

    def evaluate(self, data_loader, criterion=None, metrics=dict[str, callable]):
        return self._evaluate(
            data_loader,
            criterion=criterion,
            metrics=metrics,
            desc="Evaluating",
        )[-1]
        
    def predict(self, data_loader):
        return self._evaluate(data_loader, desc="Predicting")[0]

    def save_checkpoint(self, path: str | Path | os.PathLike):
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.state.pth")

    def load_checkpoint(self, path: str | Path | os.PathLike, device=None):
        device = self.device if device is None else device
        path = Path(path)

        if self.model is None:
            raise RuntimeError("Cannot load checkpoint without model instance.")
        self.model.load_state_dict(torch.load(path / "model.state.pth", map_location=device))
        self.model.to(device)


class Trainer(Evaulator):
    def __init__(
        self,
        model: BaseModel,
        config: BaseConfig,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        super().__init__(model, device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def _update_state(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _train_epoch(self, data_loader, criterion, desc=None):
        with tqdm(data_loader, desc=desc) as pbar:
            for batch in pbar:
                output = self.model(self._move_batch(batch))
                loss = criterion(output.pred, output.label)
                # TODO: 修改此处可以实现梯度累积、裁剪等操作，留作练习
                self._update_state(loss)
                pbar.set_postfix(train_loss=loss.item())

    def train(self, train_loader, criterion, val_loader=None, metrics=None, num_epochs=None):
        self.model.train()
        num_epochs = self.config.num_epochs if num_epochs is None else num_epochs
        for epoch in range(num_epochs):
            desc = f"Epoch {epoch+1}/{num_epochs}"
            self._train_epoch(train_loader, criterion=criterion, desc=desc)
            # TODO: 很容易修改此处，让 Trainer 保存最优的 k 的权重，以及支持早停策略，留作练习
            self._evaluate(val_loader, criterion=criterion, metrics=metrics, desc=desc)
            if self.scheduler is not None:
                self.scheduler.step()

    @override
    def save_checkpoint(self, path: str | Path | os.PathLike, only_model=True):
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.config is not None:
            self.config.save_file(path / "config.yaml")
            torch.save(self.model.state_dict(), path / "model.state.pth")
        if not only_model:
            torch.save(self.optimizer.state_dict(), path / "optimizer.state.pth")
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), path / "scheduler.state.pth")

    @override
    def load_checkpoint(self, path: str | Path | os.PathLike, device=None):
        device = self.device if device is None else device
        path = Path(path)

        if not all([self.model, self.config, self.optimizer]):
            raise RuntimeError("Cannot load checkpoint without model, config and optimizer instance.")
        self.config.load_file(path / "config.yaml")
        self.model.load_state_dict(torch.load(path / "model.state.pth", map_location=device))
        self.optimizer.load_state_dict(torch.load(path / "optimizer.state.pth", map_location=device))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(path / "scheduler.state.pth", map_location=device))
        self.model.to(device)
