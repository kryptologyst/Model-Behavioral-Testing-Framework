"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preprocessing class."""
    
    def __init__(self, dataset_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize data loader.
        
        Args:
            dataset_name: Name of the dataset to load.
            config: Optional configuration dictionary.
        """
        self.dataset_name = dataset_name
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        self.metadata = None
        
    def load_data(self, test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data.
        
        Args:
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if self.dataset_name == "iris":
            return self._load_iris(test_size, random_state)
        elif self.dataset_name == "wine":
            return self._load_wine(test_size, random_state)
        elif self.dataset_name == "breast_cancer":
            return self._load_breast_cancer(test_size, random_state)
        elif self.dataset_name == "synthetic":
            return self._load_synthetic(test_size, random_state)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_iris(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load Iris dataset."""
        data = load_iris()
        X, y = data.data, data.target
        
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.metadata = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "description": "Iris flower classification dataset"
        }
        
        return self._preprocess_data(X, y, test_size, random_state)
    
    def _load_wine(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load Wine dataset."""
        data = load_wine()
        X, y = data.data, data.target
        
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.metadata = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "description": "Wine classification dataset"
        }
        
        return self._preprocess_data(X, y, test_size, random_state)
    
    def _load_breast_cancer(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load Breast Cancer dataset."""
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.metadata = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "description": "Breast cancer classification dataset"
        }
        
        return self._preprocess_data(X, y, test_size, random_state)
    
    def _load_synthetic(self, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load synthetic dataset."""
        n_samples = self.config.get("n_samples", 1000)
        n_features = self.config.get("n_features", 10)
        n_classes = self.config.get("n_classes", 3)
        n_informative = self.config.get("n_informative", 5)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=random_state
        )
        
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
        self.target_names = [f"class_{i}" for i in range(n_classes)]
        self.metadata = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "description": "Synthetic classification dataset"
        }
        
        return self._preprocess_data(X, y, test_size, random_state)
    
    def _preprocess_data(self, X: np.ndarray, y: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data with scaling and train/test split.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data to use for testing.
            random_state: Random seed.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Loaded {self.dataset_name} dataset: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata.
        """
        return self.metadata
    
    def save_metadata(self, path: Union[str, Path]) -> None:
        """Save metadata to JSON file.
        
        Args:
            path: Path to save metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_edge_cases(self, X: np.ndarray, n_cases: int = 10) -> np.ndarray:
        """Generate edge cases for testing.
        
        Args:
            X: Original feature matrix.
            n_cases: Number of edge cases to generate.
            
        Returns:
            Array of edge cases.
        """
        edge_cases = []
        
        # Extreme values
        for _ in range(n_cases // 3):
            extreme_case = np.random.uniform(-10, 10, X.shape[1])
            edge_cases.append(extreme_case)
        
        # Zero values
        for _ in range(n_cases // 3):
            zero_case = np.zeros(X.shape[1])
            edge_cases.append(zero_case)
        
        # Random values
        for _ in range(n_cases - 2 * (n_cases // 3)):
            random_case = np.random.normal(0, 1, X.shape[1])
            edge_cases.append(random_case)
        
        return np.array(edge_cases)
    
    def generate_adversarial_cases(self, X: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Generate adversarial cases by adding small perturbations.
        
        Args:
            X: Original feature matrix.
            epsilon: Perturbation magnitude.
            
        Returns:
            Array of adversarial cases.
        """
        noise = np.random.normal(0, epsilon, X.shape)
        return X + noise
