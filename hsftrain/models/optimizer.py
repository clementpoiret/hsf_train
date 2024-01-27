import math
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("AdamP",)


def centralize_gradient(x: torch.Tensor, gc_conv_only: bool = False):
    r"""Gradient Centralization (GC).

    :param x: torch.Tensor. gradient.
    :param gc_conv_only: bool. 'False' for both conv & fc layers.
    """
    size: int = x.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))


def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):  # linear layers
        dim = 1
    elif x_len == 4:  # conv kernels
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


def agc(p: torch.Tensor,
        grad: torch.Tensor,
        agc_eps: float,
        agc_clip_val: float,
        eps: float = 1e-6) -> torch.Tensor:
    r"""Clip gradient values in excess of the unit wise norm.

    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. agc epsilon to clip the norm of parameter.
    :param agc_clip_val: float. norm clip.
    :param eps: float. simple stop from div by zero and no relation to standard optimizer eps.
    """
    p_norm = unit_norm(p).clamp_(agc_eps)
    g_norm = unit_norm(grad)

    max_norm = p_norm * agc_clip_val

    clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

    return torch.where(g_norm > max_norm, clipped_grad, grad)


class AdamP(Optimizer):
    r"""Implements AdamP algorithm.

    It has been proposed in `Slowing Down the Weight Norm Increase in
    Momentum-based Optimizers`__

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        weight_decouple: the optimizer uses decoupled weight decay as in AdamW
            (default: True)
        delta: threhold that determines whether a set of parameters is scale
            invariant or not (default: 0.1)
        wd_ratio: relative weight decay applied on scale-invariant parameters
            compared to that applied on scale-variant parameters (default: 0.1)
        use_gc: use gradient centralization (default: False)
        nesterov: enables Nesterov momentum (default: False)
        r: EMA factor. Between 0.9 ~ 0.99 is preferred. (default: 0.95)
        adanorm: use AdaNorm (default: False)
        adam_debias: only correct the denominator to avoid inflating step sizes
            early in training. (default: False)
        demon: use decayed momentum (default: False)
        epochs: number of epochs for demon (default: None)
        steps_per_epoch: number of steps per epoch for demon (default: None)
        agc_clipping_value: clipping value for adaptive gradient clipping
            (default: 1e-2)
        agc_eps: term added to the denominator to improve numerical stability
            for adaptive gradient clipping (default: 1e-3)


    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AdamP(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

     __ https://arxiv.org/abs/2006.08217

    Note:
        Reference code: https://github.com/clovaai/AdamP
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        use_gc: bool = False,
        nesterov: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        demon: bool = False,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        agc_clipping_value: float = 1e-2,
        agc_eps: float = 1e-3,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if delta < 0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if wd_ratio < 0:
            raise ValueError("Invalid wd_ratio value: {}".format(wd_ratio))
        if not 0.0 <= r < 1.0:
            raise ValueError("Invalid r value: {}".format(r))
        if demon and (epochs is None or steps_per_epoch is None):
            raise ValueError(
                "epochs and steps_per_epoch must be specified for demon")
        if agc_clipping_value <= 0:
            raise ValueError("Invalid agc_clipping_value value: {}".format(
                agc_clipping_value))
        if agc_eps <= 0:
            raise ValueError("Invalid agc_eps value: {}".format(agc_eps))

        self.use_gc = use_gc
        self.demon = demon

        if demon:
            self.T = epochs * steps_per_epoch

        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
            adanorm=adanorm,
            adam_debias=False,
        )
        if adanorm:
            defaults.update({"r": r})

        super(AdamP, self).__init__(params, defaults)

    @staticmethod
    def _channel_view(x):
        return x.view(x.size(0), -1)

    @staticmethod
    def _layer_view(x):
        return x.view(1, -1)

    @staticmethod
    def _cosine_similarity(x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad: Optional[torch.Tensor],
        lr: float,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
        ratio: Optional[float] = None,
    ):
        r"""Apply weight decay.

        :param p: torch.Tensor. parameter.
        :param grad: torch.Tensor. gradient.
        :param lr: float. learning rate.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param fixed_decay: bool. fix weight decay.
        :param ratio: Optional[float]. scale weight decay.
        """
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) *
                   (ratio if ratio is not None else 1.0))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    @staticmethod
    def get_adanorm_gradient(grad: torch.Tensor,
                             adanorm: bool,
                             exp_grad_norm: Optional[torch.Tensor] = None,
                             r: Optional[float] = 0.95) -> torch.Tensor:
        r"""Get AdaNorm gradient.

        :param grad: torch.Tensor. gradient.
        :param adanorm: bool. whether to apply AdaNorm.
        :param exp_grad_norm: Optional[torch.Tensor]. exp_grad_norm.
        :param r: float. Optional[float]. momentum (ratio).
        """
        if not adanorm:
            return grad

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul_(r).add_(grad_norm, alpha=1.0 - r)

        return grad * exp_grad_norm / grad_norm if exp_grad_norm > grad_norm else grad

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float,
                          bias_correction1: float) -> float:
        r"""Apply AdamD variant.

        :param adam_debias: bool. whether to apply AdamD.
        :param step_size: float. step size.
        :param bias_correction1: float. bias_correction.
        """
        return step_size if adam_debias else step_size / bias_correction1

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(
                    p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(
                    p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                beta1_init, beta2 = group["betas"]

                # Apply Demon decay
                if self.demon:
                    temp = 1 - (group["step"] / self.T)
                    beta1 = beta1_init * temp / (
                        (1 - beta1_init) + beta1_init * temp)
                else:
                    beta1 = beta1_init

                nesterov = group["nesterov"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    if group["adanorm"]:
                        state["exp_grad_norm"] = torch.zeros((1,),
                                                             dtype=grad.dtype,
                                                             device=grad.device)

                # Apply Adaptive Gradient Clipping (AGC)
                grad.copy_(
                    agc(p.data, grad, self.agc_eps, self.agc_clipping_value))

                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group["adanorm"],
                    exp_grad_norm=state.get("exp_grad_norm", None),
                    r=group.get("r", None),
                )

                # Adam
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1**state["step"]
                bias_correction2 = 1 - beta2**state["step"]

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"])
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p,
                        grad,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # Weight decay
                self.apply_weight_decay(
                    p=p.data,
                    grad=None,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    weight_decouple=group["weight_decouple"],
                    fixed_decay=group["fixed_decay"],
                    ratio=wd_ratio,
                )

                # Step
                step_size = self.apply_adam_debias(
                    adam_debias=group["adam_debias"],
                    step_size=group["lr"],
                    bias_correction1=bias_correction1,
                )
                p.data.add_(perturb, alpha=-step_size)

        return loss
