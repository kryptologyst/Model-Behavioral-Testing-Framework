"""Test suite for the model behavioral testing framework."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src import DataLoader, ModelFactory, BehavioralTester, set_seed, get_device


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader("iris")
        assert loader.dataset_name == "iris"
        assert loader.config == {}
    
    def test_load_iris_data(self):
        """Test loading Iris dataset."""
        loader = DataLoader("iris")
        X_train, X_test, y_train, y_test = loader.load_data()
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert X_train.shape[1] == 4  # Iris has 4 features
    
    def test_load_synthetic_data(self):
        """Test loading synthetic dataset."""
        config = {"n_samples": 100, "n_features": 5, "n_classes": 2}
        loader = DataLoader("synthetic", config)
        X_train, X_test, y_train, y_test = loader.load_data()
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == 5
        assert len(np.unique(y_train)) == 2
    
    def test_generate_edge_cases(self):
        """Test edge case generation."""
        loader = DataLoader("iris")
        X_train, _, _, _ = loader.load_data()
        
        edge_cases = loader.generate_edge_cases(X_train, n_cases=5)
        assert edge_cases.shape[0] == 5
        assert edge_cases.shape[1] == X_train.shape[1]
    
    def test_generate_adversarial_cases(self):
        """Test adversarial case generation."""
        loader = DataLoader("iris")
        X_train, _, _, _ = loader.load_data()
        
        adversarial_cases = loader.generate_adversarial_cases(X_train, epsilon=0.1)
        assert adversarial_cases.shape == X_train.shape


class TestModelFactory:
    """Test cases for ModelFactory."""
    
    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        model = ModelFactory.create_model("random_forest")
        assert isinstance(model, RandomForestClassifier)
    
    def test_create_model_with_config(self):
        """Test creating model with custom config."""
        config = {"n_estimators": 50, "max_depth": 5}
        model = ModelFactory.create_model("random_forest", config)
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    def test_get_available_models(self):
        """Test getting available model types."""
        models = ModelFactory.get_available_models()
        assert "random_forest" in models
        assert "logistic_regression" in models
        assert len(models) > 0
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = ModelFactory.get_default_config("random_forest")
        assert "n_estimators" in config
        assert "random_state" in config


class TestBehavioralTester:
    """Test cases for BehavioralTester."""
    
    def setup_method(self):
        """Setup test data and model."""
        # Create synthetic data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Split data
        self.X_test = X[:30]
        self.y_test = y[:30]
        
        # Create tester
        self.tester = BehavioralTester(self.model)
    
    def test_behavioral_tester_initialization(self):
        """Test BehavioralTester initialization."""
        assert self.tester.model is not None
        assert len(self.tester.tests) > 0
        assert len(self.tester.results) == 0
    
    def test_run_comprehensive_tests(self):
        """Test running comprehensive tests."""
        results = self.tester.run_comprehensive_tests(self.X_test, self.y_test)
        
        assert "overall" in results
        assert "edge_case_test" in results
        assert "robustness_test" in results
        assert "calibration_test" in results
        
        assert results["overall"]["n_tests"] > 0
        assert results["overall"]["n_passed"] >= 0
    
    def test_run_specific_test(self):
        """Test running specific test."""
        result = self.tester.run_specific_test("edge_case_test", self.X_test, self.y_test)
        
        assert result.test_name == "edge_case_test"
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)
        assert isinstance(result.details, dict)
    
    def test_get_summary(self):
        """Test getting test summary."""
        # Run tests first
        self.tester.run_comprehensive_tests(self.X_test, self.y_test)
        
        summary = self.tester.get_summary()
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "overall_score" in summary


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert hasattr(device, 'type')


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Load data
        loader = DataLoader("iris")
        X_train, X_test, y_train, y_test = loader.load_data()
        
        # Create and train model
        model = ModelFactory.create_model("random_forest")
        model.fit(X_train, y_train)
        
        # Run behavioral tests
        tester = BehavioralTester(model)
        results = tester.run_comprehensive_tests(X_test, y_test)
        
        # Verify results
        assert "overall" in results
        assert results["overall"]["n_tests"] > 0
        
        # Test summary
        summary = tester.get_summary()
        assert summary["total_tests"] > 0
    
    def test_multiple_models_comparison(self):
        """Test comparing multiple models."""
        loader = DataLoader("iris")
        X_train, X_test, y_train, y_test = loader.load_data()
        
        models = ["random_forest", "logistic_regression"]
        results = {}
        
        for model_type in models:
            model = ModelFactory.create_model(model_type)
            model.fit(X_train, y_train)
            
            tester = BehavioralTester(model)
            test_results = tester.run_comprehensive_tests(X_test, y_test)
            results[model_type] = test_results["overall"]["score"]
        
        assert len(results) == 2
        assert all(isinstance(score, float) for score in results.values())


if __name__ == "__main__":
    pytest.main([__file__])
