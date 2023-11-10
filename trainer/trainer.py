import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import jit
import wandb


class Trainer:
    def train(self, model, optimizer, loss_fn, train_loader, val_loader, lr_shedule, epochs=20, device='cpu', acc_steps=32):
        step = 0
        model = model.to(device).to(torch.float16)

        optimizer.zero_grad()
        for epoch in range(epochs):
            for i, subject in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                X = torch.stack(
                    [i['data'] for k, i in subject.items() if k != 'label']).squeeze(2).permute(1, 0, 2, 3, 4).to(device).to(torch.float16)
                y = subject['label']['data'].squeeze(
                    2).permute(1, 0, 2, 3, 4).to(device).to(torch.float16)

                output = model(X)
                loss = loss_fn(output, y)
                loss /= acc_steps
                loss.backward()
                wandb.log({"loss": loss})
                step += 1

                if i % acc_steps == 0 or i == len(train_loader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()

            if lr_shedule is not None:
                lr_shedule.step()

        return model
