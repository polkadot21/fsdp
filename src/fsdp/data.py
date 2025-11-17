import torch


def make_batch(batch_size: int, T, D, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, T, D, device=device)
    y = x.clone()
    return x, y


def cpu_cfg():
    return dict(
        in_dim=128,
        dim=256,
        n_heads=8,
        ff_dim=1024,
        n_layers=4,
        batch=4,
        T=64,
        steps=50,
        lr=1e-3,
        wd=0.0,
    )


def cloud_cfg():  # "fat" to make comm/compute meaningful on H100
    return dict(
        in_dim=2048,
        dim=4096,
        n_heads=32,
        ff_dim=16384,
        n_layers=8,
        batch=4,
        T=256,
        steps=100,
        lr=1e-3,
        wd=0.0,
    )
