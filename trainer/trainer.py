from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb

from losses.loss_wrapper import LossWrapper


class Trainer:
    def __init__(self, n_classes: int = 5, segmentation_log_budget: int = 3) -> None:
        self._classes = list(range(n_classes))
        self._segmentation_log_budget = segmentation_log_budget

    def train(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: LossWrapper, train_loader: DataLoader, val_loader: DataLoader, lr_shedule: Optional[optim.lr_scheduler.LRScheduler], epochs: int = 20, device: str = 'cpu', acc_steps: int = 32):
        step = 0
        model = model.to(torch.float16).to(device)

        wandb.init(project="ml4mip")
        wandb.watch(model, log="all", log_freq=100)

        optimizer.zero_grad()
        for epoch in range(epochs):
            step = self._do_phase(model, train_loader, loss_fn,
                                  optimizer, device, acc_steps, train=True, step=step)
            step = self._do_phase(
                model, val_loader, loss_fn, optimizer, device, acc_steps, train=False, step=step)

            if lr_shedule is not None:
                lr_shedule.step()

        wandb.finish()

        return model

    def __format_tensor(self, tensor: torch.Tensor, device: torch.device):
        return tensor.squeeze(2).permute(1, 0, 2, 3, 4).to(torch.float16).to(device)

    def _do_phase(self, model: nn.Module, loader: DataLoader, criterion: LossWrapper, optimizer: optim.Optimizer, device: torch.device, acc_steps: int, train: bool = True, step: int = 0, metrics: list[nn.Module] = None):
        if train:
            model.train()
        else:
            model.eval()

        prefix = "train" if train else "val"
        criterion.set_prefix(prefix + "/")
        seg_buget = self._segmentation_log_budget

        optimizer.zero_grad()
        for i, subject in enumerate(loader):
            X = self.__format_tensor(torch.stack(
                [i['data'] for k, i in subject.items() if k != 'label']), device=device)
            y = self.__format_tensor(subject['label']['data'], device=device)

            output = model(X)
            criterion.set_step(step)
            loss = criterion(output, y)
            loss /= acc_steps

            if train:
                loss.backward()

            computed_metrics = {}
            if metrics is not None:
                computed_metrics = {f"{prefix}/{metric.__class__.__name__}": metric(output, y)
                                    for metric in metrics}

            wandb.log({f"{prefix}/loss": loss, **computed_metrics}, step=step)

            if not train and seg_buget > 0:
                shape = torch.as_tensor(X[0].shape) // 2
                segmentation = wandb.Image(
                    X[0, *shape].cpu().numpy(),
                    masks={
                        "ground_truth": {
                            "mask_data": y[0, *shape].cpu().numpy(),
                            "class_labels": self._classes
                        },
                        "predictions": {
                            "mask_data": output[0, *shape].cpu().numpy(),
                            "class_labels": self._classes
                        },
                    },
                    caption="Segmentation")
                wandb.log({f"{prefix}/segmentation": segmentation}, step=step)
                seg_buget -= 1

            step += 1

            if i % acc_steps == 0 or i == len(loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        return step
