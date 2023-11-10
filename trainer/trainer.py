import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import jit


class Trainer:
    def __init__(self) -> None:
        self._summary_writer = SummaryWriter()

    def train(self, model, optimizer, loss_fn, train_loader, val_loader, lr_shedule, epochs=20, device='cpu', acc_steps=32):
        step = 0
        model = model.to(device).to(torch.float16)

        optimizer.zero_grad()
        for epoch in tqdm(range(epochs)):
            for i, subject in enumerate(train_loader):
                X = torch.stack(
                    [i['data'] for k, i in subject.items() if k != 'label']).squeeze(2).permute(1, 0, 2, 3, 4).to(device).to(torch.float16)
                y = subject['label']['data'].squeeze(
                    2).permute(1, 0, 2, 3, 4).to(device).to(torch.float16)

                output = model(X)
                loss = loss_fn(output, y)
                loss /= acc_steps
                loss.backward()
                self._summary_writer.add_scalar(
                    'Loss/train', loss.item(), step)
                step += 1

                if i % acc_steps == 0 or i == len(train_loader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()

            if lr_shedule is not None:
                lr_shedule.step()

        return model
