import torch
import torch.nn as nn

from src.optimizers.entropy_sgd import EntropySGD


def initialize_optimizer(model: nn.Module, optimizer_name: str = "SGD", **kwargs):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "Entropy-SGD":
        return EntropySGD(model.parameters(), kwargs)
    else:
        raise ValueError("Optimizer '{}' not recognized.".format(optimizer_name))
