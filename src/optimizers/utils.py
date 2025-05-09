import torch
import torch.nn as nn

from src.optimizers.c_flat import C_Flat
from src.optimizers.entropy_sgd import EntropySGD
from src.optimizers.second_order_optimizer import SecondOrderOptimizer


def initialize_optimizer(model: nn.Module, optimizer_name: str = "SGD", **kwargs):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "Entropy-SGD":
        return EntropySGD(model.parameters(), kwargs)
    elif optimizer_name == "C-Flat":
        return C_Flat(model.parameters(), model=model, cflat=True, **kwargs)
    elif optimizer_name == "Second-Order":
        return SecondOrderOptimizer(model.parameters(), model=model, **kwargs)
    else:
        raise ValueError("Optimizer '{}' not recognized.".format(optimizer_name))
