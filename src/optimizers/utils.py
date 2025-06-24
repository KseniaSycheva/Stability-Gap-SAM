import torch
import torch.nn as nn

from src.optimizers.c_flat import C_Flat
from src.optimizers.entropy_sgd import EntropySGD


def initialize_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_name: str = "SGD",
    base_optimizer=None,
    **kwargs,
):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "Entropy-SGD":
        return EntropySGD(
            model.parameters(), base_optimizer=base_optimizer, config=kwargs
        )
    elif optimizer_name == "C-Flat":
        return C_Flat(
            model.parameters(),
            lr=lr,
            base_optimizer=base_optimizer,
            model=model,
            cflat=True,
            **kwargs,
        )
    else:
        raise ValueError("Optimizer '{}' not recognized.".format(optimizer_name))
