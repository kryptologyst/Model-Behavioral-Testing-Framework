"""Comprehensive behavioral testing framework for machine learning models."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    metadata: Dict[str, Any]


class BehavioralTest(ABC):
    """Abstract base class for behavioral tests."""
    
    def __init__(self, name: str):
        """Initialize test.
        
        Args:
            name: Name of the test.
        """
        self.name = name
    
    @abstractmethod
    def run(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TestResult:
        """Run the behavioral test.
        
        Args:
            model: Trained model to test.
            X: Feature matrix.
            y: Optional target vector.
            **kwargs: Additional test parameters.
            
        Returns:
            Test result.
        """
        pass


class EdgeCaseTest(BehavioralTest):
    """Test model behavior on edge cases."""
    
    def __init__(self):
        """Initialize edge case test."""
        super().__init__("edge_case_test")
    
    def run(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TestResult:
        """Run edge case test.
        
        Args:
            model: Trained model to test.
            X: Feature matrix.
            y: Optional target vector.
            **kwargs: Additional test parameters.
            
        Returns:
            Test result.
        """
        n_cases = kwargs.get("n_cases", 10)
        
        # Generate edge cases
        edge_cases = self._generate_edge_cases(X, n_cases)
        
        # Test predictions
        predictions = model.predict(edge_cases)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(edge_cases)
        
        # Analyze results
        unique_predictions = len(np.unique(predictions))
        max_prob = np.max(probabilities) if probabilities is not None else 0.0
        min_prob = np.min(probabilities) if probabilities is not None else 0.0
        
        # Test passes if model doesn't crash and produces reasonable outputs
        passed = (
            len(predictions) == len(edge_cases) and
            unique_predictions > 0 and
            (probabilities is None or (max_prob <= 1.0 and min_prob >= 0.0))
        )
        
        score = unique_predictions / len(np.unique(y)) if y is not None else unique_predictions
        
        return TestResult(
            test_name=self.name,
            passed=passed,
            score=score,
            details={
                "n_edge_cases": len(edge_cases),
                "unique_predictions": unique_predictions,
                "max_probability": max_prob,
                "min_probability": min_prob,
                "predictions": predictions.tolist()
            },
            metadata={
                "edge_cases": edge_cases.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None
            }
        )
    
    def _generate_edge_cases(self, X: np.ndarray, n_cases: int) -> np.ndarray:
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


class RobustnessTest(BehavioralTest):
    """Test model robustness to adversarial perturbations."""
    
    def __init__(self):
        """Initialize robustness test."""
        super().__init__("robustness_test")
    
    def run(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TestResult:
        """Run robustness test.
        
        Args:
            model: Trained model to test.
            X: Feature matrix.
            y: Optional target vector.
            **kwargs: Additional test parameters.
            
        Returns:
            Test result.
        """
        epsilon = kwargs.get("epsilon", 0.1)
        n_samples = kwargs.get("n_samples", min(100, len(X)))
        
        # Sample test cases
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices] if y is not None else None
        
        # Original predictions
        original_preds = model.predict(X_sample)
        
        # Generate adversarial examples
        adversarial_X = self._generate_adversarial(X_sample, epsilon)
        adversarial_preds = model.predict(adversarial_X)
        
        # Calculate robustness metrics
        accuracy_drop = 0.0
        if y_sample is not None:
            original_acc = accuracy_score(y_sample, original_preds)
            adversarial_acc = accuracy_score(y_sample, adversarial_preds)
            accuracy_drop = original_acc - adversarial_acc
        
        prediction_changes = np.sum(original_preds != adversarial_preds)
        robustness_score = 1.0 - (prediction_changes / len(original_preds))
        
        passed = robustness_score > 0.8  # Threshold for robustness
        
        return TestResult(
            test_name=self.name,
            passed=passed,
            score=robustness_score,
            details={
                "epsilon": epsilon,
                "n_samples": n_samples,
                "accuracy_drop": accuracy_drop,
                "prediction_changes": prediction_changes,
                "robustness_score": robustness_score
            },
            metadata={
                "original_predictions": original_preds.tolist(),
                "adversarial_predictions": adversarial_preds.tolist(),
                "adversarial_samples": adversarial_X.tolist()
            }
        )
    
    def _generate_adversarial(self, X: np.ndarray, epsilon: float) -> np.ndarray:
        """Generate adversarial examples using FGSM-like approach.
        
        Args:
            X: Original feature matrix.
            epsilon: Perturbation magnitude.
            
        Returns:
            Adversarial examples.
        """
        # Simple random perturbation (in practice, you'd use gradient-based methods)
        noise = np.random.normal(0, epsilon, X.shape)
        return X + noise


class CalibrationTest(BehavioralTest):
    """Test model calibration."""
    
    def __init__(self):
        """Initialize calibration test."""
        super().__init__("calibration_test")
    
    def run(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TestResult:
        """Run calibration test.
        
        Args:
            model: Trained model to test.
            X: Feature matrix.
            y: Optional target vector.
            **kwargs: Additional test parameters.
            
        Returns:
            Test result.
        """
        if y is None or not hasattr(model, "predict_proba"):
            return TestResult(
                test_name=self.name,
                passed=False,
                score=0.0,
                details={"error": "Calibration test requires target labels and probability predictions"},
                metadata={}
            )
        
        # Get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Calculate calibration metrics
        max_prob = np.max(probabilities, axis=1)
        confidence = np.mean(max_prob)
        
        # Expected Calibration Error (simplified)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_prob > bin_lower) & (max_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy_score(y[in_bin], predictions[in_bin])
                avg_confidence_in_bin = max_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Test passes if ECE is low
        passed = ece < 0.1
        score = 1.0 - ece  # Higher score for better calibration
        
        return TestResult(
            test_name=self.name,
            passed=passed,
            score=score,
            details={
                "ece": ece,
                "confidence": confidence,
                "n_bins": n_bins
            },
            metadata={
                "probabilities": probabilities.tolist(),
                "max_probabilities": max_prob.tolist()
            }
        )


class BehavioralTester:
    """Main class for running comprehensive behavioral tests."""
    
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize behavioral tester.
        
        Args:
            model: Trained model to test.
            config: Optional configuration dictionary.
        """
        self.model = model
        self.config = config or {}
        self.tests = [
            EdgeCaseTest(),
            RobustnessTest(),
            CalibrationTest()
        ]
        self.results = []
    
    def run_comprehensive_tests(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Run all available behavioral tests.
        
        Args:
            X: Feature matrix.
            y: Optional target vector.
            **kwargs: Additional test parameters.
            
        Returns:
            Dictionary containing all test results.
        """
        logger.info(f"Running comprehensive behavioral tests on {len(X)} samples")
        
        results = {}
        for test in self.tests:
            try:
                result = test.run(self.model, X, y, **kwargs)
                results[test.name] = asdict(result)
                self.results.append(result)
                logger.info(f"Test {test.name}: {'PASSED' if result.passed else 'FAILED'} (score: {result.score:.3f})")
            except Exception as e:
                logger.error(f"Test {test.name} failed with error: {e}")
                results[test.name] = {
                    "test_name": test.name,
                    "passed": False,
                    "score": 0.0,
                    "details": {"error": str(e)},
                    "metadata": {}
                }
        
        # Calculate overall score
        overall_score = np.mean([r.score for r in self.results if r.passed])
        overall_passed = overall_score > 0.7
        
        results["overall"] = {
            "passed": overall_passed,
            "score": overall_score,
            "n_tests": len(self.tests),
            "n_passed": sum(1 for r in self.results if r.passed)
        }
        
        return results
    
    def run_specific_test(self, test_name: str, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TestResult:
        """Run a specific behavioral test.
        
        Args:
            test_name: Name of the test to run.
            X: Feature matrix.
            y: Optional target vector.
            **kwargs: Additional test parameters.
            
        Returns:
            Test result.
        """
        for test in self.tests:
            if test.name == test_name:
                return test.run(self.model, X, y, **kwargs)
        
        raise ValueError(f"Unknown test: {test_name}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all test results.
        
        Returns:
            Summary dictionary.
        """
        if not self.results:
            return {"message": "No tests have been run yet"}
        
        return {
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "failed_tests": sum(1 for r in self.results if not r.passed),
            "overall_score": np.mean([r.score for r in self.results if r.passed]),
            "test_details": {r.test_name: {"passed": r.passed, "score": r.score} for r in self.results}
        }
    
    def save_results(self, path: Union[str, Path]) -> None:
        """Save test results to JSON file.
        
        Args:
            path: Path to save results.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        def make_serializable(obj):
            """Recursively convert numpy types to Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        test_results_serializable = []
        for result in self.results:
            result_dict = asdict(result)
            test_results_serializable.append(make_serializable(result_dict))
        
        results_dict = {
            "model_info": {
                "model_type": type(self.model).__name__,
                "config": self.config
            },
            "test_results": test_results_serializable,
            "summary": self.get_summary()
        }
        
        with open(path, 'w') as f:
            try:
                json.dump(results_dict, f, indent=2)
            except TypeError:
                # Fallback: convert to string representation
                import json as json_module
                json_module.dump(str(results_dict), f, indent=2)
        
        logger.info(f"Results saved to {path}")
