import torch


def make_batch(batch_size: int, T, D, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, T, D, device=device)
    y = x.clone()
    return x, y
