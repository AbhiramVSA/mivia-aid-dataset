from __future__ import annotations

import random
from collections.abc import Callable

import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def _seed_worker(worker_id: int) -> None:
        worker_seed = (base_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, min(warmup_epochs, num_epochs))

    def _lr_lambda(epoch_index: int) -> float:
        if num_epochs <= 1:
            return 1.0
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        if epoch_index >= num_epochs - 1:
            return min_lr_ratio
        cosine_epochs = max(1, num_epochs - warmup_epochs)
        progress = float(epoch_index - warmup_epochs) / float(max(1, cosine_epochs - 1))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
