"""Model factory for creating different types of models."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating different types of models."""
    
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a model instance.
        
        Args:
            model_type: Type of model to create.
            config: Optional configuration dictionary.
            
        Returns:
            Model instance.
        """
        config = config or {}
        
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", None),
                random_state=config.get("random_state", 42),
                n_jobs=config.get("n_jobs", -1)
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=config.get("n_estimators", 100),
                learning_rate=config.get("learning_rate", 0.1),
                max_depth=config.get("max_depth", 3),
                random_state=config.get("random_state", 42)
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                random_state=config.get("random_state", 42),
                max_iter=config.get("max_iter", 1000),
                C=config.get("C", 1.0)
            )
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=config.get("max_depth", None),
                random_state=config.get("random_state", 42)
            )
        elif model_type == "svm":
            return SVC(
                kernel=config.get("kernel", "rbf"),
                C=config.get("C", 1.0),
                probability=True,
                random_state=config.get("random_state", 42)
            )
        elif model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=config.get("hidden_layer_sizes", (100,)),
                max_iter=config.get("max_iter", 1000),
                random_state=config.get("random_state", 42)
            )
        elif model_type == "calibrated_logistic":
            base_model = LogisticRegression(random_state=config.get("random_state", 42))
            return CalibratedClassifierCV(base_model, cv=config.get("cv", 3))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available model types.
        
        Returns:
            List of available model types.
        """
        return [
            "random_forest",
            "gradient_boosting", 
            "logistic_regression",
            "decision_tree",
            "svm",
            "mlp",
            "calibrated_logistic"
        ]
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type.
        
        Args:
            model_type: Type of model.
            
        Returns:
            Default configuration dictionary.
        """
        configs = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42,
                "n_jobs": -1
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "logistic_regression": {
                "random_state": 42,
                "max_iter": 1000,
                "C": 1.0
            },
            "decision_tree": {
                "max_depth": None,
                "random_state": 42
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "probability": True,
                "random_state": 42
            },
            "mlp": {
                "hidden_layer_sizes": (100,),
                "max_iter": 1000,
                "random_state": 42
            },
            "calibrated_logistic": {
                "random_state": 42,
                "cv": 3
            }
        }
        
        return configs.get(model_type, {})
