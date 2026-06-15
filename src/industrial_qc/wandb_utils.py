from __future__ import annotations

import os
import warnings
from collections.abc import Mapping


def login_wandb(api_key: str | None = None) -> bool:
    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"W&B import skipped: {exc}")
        return False

    resolved_key = api_key or os.getenv("WANDB_API_KEY")
    if not resolved_key:
        warnings.warn("WANDB_API_KEY is not set; continuing without W&B login.")
        return False

    try:
        wandb.login(key=resolved_key, relogin=True)
        return True
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"W&B login failed: {exc}")
        return False


def maybe_init_wandb(project: str, name: str, config: Mapping[str, object], api_key: str | None = None):
    try:
        import wandb
    except Exception:
        return None

    if not login_wandb(api_key=api_key):
        return None

    try:
        return wandb.init(project=project, name=name, config=dict(config))
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"W&B init failed: {exc}")
        return None
