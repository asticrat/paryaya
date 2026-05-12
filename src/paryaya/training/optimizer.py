"""
paryaya.training.optimizer — Noam learning-rate schedule + Adam.

Schedule (Vaswani et al. 2017):
    lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

Peaks at step == warmup_steps, then decays as inverse-square-root.
Step is 1-indexed to avoid division-by-zero at the very first update.

Usage:
    opt, sched = get_noam_optimizer(model.parameters(), d_model=512, warmup_steps=4000)
    ...
    loss.backward()
    opt.step(); sched.step()
    current_lr = sched.get_last_lr()[0]
"""
import torch


def get_noam_optimizer(
    params,
    d_model: int = 512,
    warmup_steps: int = 4000,
    factor: float = 1.0,
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.LambdaLR]:
    """Return (Adam optimizer, Noam LambdaLR scheduler) sharing the same parameter group."""
    opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        return factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched
