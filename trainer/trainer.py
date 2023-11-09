from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(self) -> None:
        self._summary_writer = SummaryWriter()

    def train(self, model, optimizer, loss_fn, train_loader, val_loader, lr_shedule, epochs=20, device='cpu'):
        step = 0
        for epoch in tqdm(range(epochs)):
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                output = model(X)
                loss = loss_fn(output, y)
                self._summary_writer.add_scalar(
                    'Loss/train', loss.item(), step)

                loss.backward()
                optimizer.step()
                step += 1

            if lr_shedule is not None:
                lr_shedule.step()

        return model
