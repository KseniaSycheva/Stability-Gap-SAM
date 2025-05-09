import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torch.optim import Optimizer


def second_order_helper(
    model,
    optimizer,
    data_loader,
    criterion,
    lamb=5e-5,
    clip_value=0.2,
    **kwargs,
):
    def closure():
        x, y = next(data_loader)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        x, y = Variable(x), Variable(y.squeeze())

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(
            input=y_hat,
            target=y,
        )

        log_probs = F.log_softmax(y_hat, dim=1)
        log_probs_reduced = 0.0
        for i in range(log_probs.shape[0]):
            log_probs_reduced += log_probs[i, y[i]]

        grad_log_probs = torch.autograd.grad(
            log_probs_reduced, model.parameters(), create_graph=True
        )

        clipped_grads = [
            torch.clamp(g, min=-clip_value, max=clip_value) for g in grad_log_probs
        ]
        fisher_penalty = sum([g.pow(2).sum() for g in clipped_grads])

        print(fisher_penalty * lamb)

        total_loss = loss + lamb * fisher_penalty

        total_loss.backward()

        return total_loss.data

    return closure


class SecondOrderOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, model, data_loader, criterion, **kwargs):
        defaults = dict(**kwargs)
        super().__init__(params, defaults)

        self.param = params
        self.base_optimizer = base_optimizer

        self.closure = second_order_helper(
            model, base_optimizer, data_loader, criterion
        )
        self.data_len = len(data_loader)

        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion

    def step(self, closure=None):
        return self.base_optimizer.step(closure=closure)
