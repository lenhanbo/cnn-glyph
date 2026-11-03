"""Stage transition and complete training-state checkpoint helpers."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch


def load_checkpoint_file(path, map_location=None):
    """Load trusted training state on both old and new PyTorch releases."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 1.13 used by the original VATr environment
        return torch.load(path, map_location=map_location)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def switch_generator_to_stage2(generator, optimizer: torch.optim.Optimizer):
    """Switch query backend and retain Adam state for every surviving named parameter."""
    before = dict(generator.named_parameters())
    old_state_by_name = {
        name: copy.deepcopy(optimizer.state[param]) for name, param in before.items() if param in optimizer.state
    }
    generator.query_embed.switch_to_stage2()
    new_optimizer = type(optimizer)(generator.parameters(), **optimizer.defaults)
    for name, param in generator.named_parameters():
        if name in old_state_by_name:
            new_optimizer.state[param] = old_state_by_name[name]
    return new_optimizer


def checkpoint_payload(model, epoch: int, global_step: int, scaler=None, scheduler=None, **extra) -> dict:
    payload = {
        "model": model.state_dict(), "epoch": epoch, "global_step": global_step,
        "optimizers": [optimizer.state_dict() for optimizer in model.optimizers],
        "rng_state": capture_rng_state(), "query_mode": model.netG.query_embed.mode,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload.update(extra)
    return payload


def save_checkpoint(path: str | Path, *args, **kwargs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(*args, **kwargs), path)
