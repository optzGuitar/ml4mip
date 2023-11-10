import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb


class Trainer:
    def train(self, model, optimizer, loss_fn, train_loader, val_loader, lr_shedule, epochs=20, device='cpu', acc_steps=32):
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

    def _do_phase(self, model: nn.Module, loader: DataLoader, criteriom: nn.Module, optimizer: optim.Optimizer, device: torch.device, acc_steps: int, train: bool = True, step: int = 0, metrics: list[nn.Module] = None):
        if train:
            model.train()
        else:
            model.eval()

        prefix = "train" if train else "val"

        optimizer.zero_grad()
        for i, subject in enumerate(loader):
            X = self.__format_tensor(torch.stack(
                [i['data'] for k, i in subject.items() if k != 'label']), device=device)
            y = self.__format_tensor(subject['label']['data'], device=device)

            output = model(X)
            loss = criteriom(output, y)
            loss /= acc_steps

            if train:
                loss.backward()

            computed_metrics = {}
            if metrics is not None:
                computed_metrics = {f"{prefix}/{metric.__class__.__name__}": metric(output, y)
                                    for metric in metrics}

            wandb.log({f"{prefix}/loss": loss, **computed_metrics}, step=step)

            step += 1

            if i % acc_steps == 0 or i == len(loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

        return step
