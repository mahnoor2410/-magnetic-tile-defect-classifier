"""
Shared configuration loader.

Every script in this project (training, evaluation, inference, tests) loads
its settings through this single function so there is exactly one source of
truth for paths, hyperparameters, and class names.
"""

import os
import random

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard requirement, but
    torch = None      # this keeps the loader importable in lightweight contexts.


def get_project_root() -> str:
    """Return the absolute path to the project root (parent of `src/`)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = None) -> dict:
    """
    Load the YAML configuration file and resolve all relative paths to
    absolute paths based on the project root, so scripts behave the same
    regardless of the working directory they are launched from.
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    root = get_project_root()
    config["_project_root"] = root

    # Resolve data / output paths to absolute paths.
    config["data"]["raw_dir"] = os.path.join(root, config["data"]["raw_dir"])
    config["model"]["checkpoint_dir"] = os.path.join(root, config["model"]["checkpoint_dir"])
    config["evaluation"]["metrics_dir"] = os.path.join(root, config["evaluation"]["metrics_dir"])
    config["evaluation"]["figures_dir"] = os.path.join(root, config["evaluation"]["figures_dir"])
    config["logging"]["log_dir"] = os.path.join(root, config["logging"]["log_dir"])
    config["inference"]["checkpoint_path"] = os.path.join(root, config["inference"]["checkpoint_path"])

    return config


def set_global_seed(seed: int = 42) -> None:
    """Set every relevant RNG seed so runs are reproducible end to end."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
