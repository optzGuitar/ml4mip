import os
import torch as th
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


class TrainLoop():
    def __init__(self, model, dataloader, loss_fn, optimizer) -> None:
        self.model = model
        self.model.cuda(1)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
    
    def __format_tensor(self, tensor: th.Tensor, device: th.device):
        return tensor.squeeze(2).permute(1, 0, 2, 3, 4).to(th.float16).to(device)
    
    def loop(self, epochs):
        for i in range(epochs):
            size = len(self.dataloader.dataset)
            self.model.train()
            for batch in tqdm(self.dataloader):
                X = self.__format_tensor(th.stack(
                    [i['data'] for k, i in batch.items() if k != 'label']), device=self.device)

                # Compute prediction error
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

