"""Test configuration for pytest."""

import pytest
import numpy as np
from src import set_seed


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with deterministic seeding."""
    set_seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_informative=3,
        random_state=42
    )
    
    return X, y


@pytest.fixture
def sample_model():
    """Provide a trained sample model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, X, y
