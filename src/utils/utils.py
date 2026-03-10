"""Utility functions for the model behavioral testing framework."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
import logging
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    import yaml
    
    ensure_dir(Path(config_path).parent)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class Config:
    """Configuration management class."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary.
        """
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary.
        """
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates.
        """
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Config):
                if isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)


def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate and clean input data.
    
    Args:
        X: Feature matrix.
        y: Optional target vector.
        
    Returns:
        Validated and cleaned data.
        
    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
    
    return X, y


def compute_feature_stats(X: np.ndarray) -> Dict[str, Any]:
    """Compute basic statistics for features.
    
    Args:
        X: Feature matrix.
        
    Returns:
        Dictionary containing feature statistics.
    """
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0),
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0),
        "median": np.median(X, axis=0),
        "q25": np.percentile(X, 25, axis=0),
        "q75": np.percentile(X, 75, axis=0),
    }
