import torch
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm

import contextlib


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class C_Flat(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        model,
        cflat=False,
        rho=0.2,
        lamb=0.2,
        adaptive=False,
        perturb_eps=1e-12,
        grad_reduce="mean",
        **kwargs,
    ):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(C_Flat, self).__init__(params, defaults)
        self.perturb_eps = perturb_eps
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.get_grad_reduce(grad_reduce)

        self.cflat = cflat
        self.rho = rho
        self.lamb = lamb

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == "mean":
            if hasattr(ReduceOp, "AVG"):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == "sum":
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.rho / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)  # e_w
                    p.add_(e_w)  # w + e_w
                    self.state[p]["e_w_0"] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)
                    self.state[p]["e_w_1_2"] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]["e_w_1_2"] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key: str):
        for group in self.param_groups:
            for p in group["params"]:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    @torch.no_grad()
    def gradient_aggregation(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data = self.state[p]["g_1"] + self.lamb * (
                    p.grad.data.detach().clone() - self.state[p]["g_2"]
                )

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False):
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def base_step(self):
        self.base_optimizer.step()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                loss_list = loss_fn()
                total_loss = sum(loss_list)
            total_loss.backward()
            return loss_list

        self.forward_backward_func = get_grad

    def step(self, delay=False, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            loss_list = get_grad()

            if self.cflat:
                self.perturb_weights(perturb_idx=0)

                disable_running_stats(self.model)
                get_grad()

                self.unperturb(perturb_key="e_w_0")
                self.grad_norm_ascent()
                get_grad()

                self.perturb_weights(perturb_idx=1)
                get_grad()

                self.gradient_aggregation()

                self.unperturb(perturb_key="e_w_1_2")

        self._sync_grad()

        if not delay:
            self.base_optimizer.step()

        enable_running_stats(self.model)

        return loss_list
