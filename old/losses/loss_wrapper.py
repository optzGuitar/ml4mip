import torch.nn as nn


class LossWrapper(nn.Module):
    def __init__(self, log_individual_losses: bool) -> None:
        super().__init__()
        self._log_individual_losses = log_individual_losses

    def set_prefix(self, prefix: str):
        self._prefix = prefix

    def set_step(self, step: int):
        self._step = step
