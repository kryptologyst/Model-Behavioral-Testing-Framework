"""Evaluation metrics and leaderboard for behavioral testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    test_name: str
    accuracy: float
    robustness_score: float
    calibration_error: float
    edge_case_performance: float
    overall_score: float
    metadata: Dict[str, Any]


class MetricsCalculator:
    """Calculate various evaluation metrics for behavioral testing."""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Accuracy score.
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_robustness_score(original_acc: float, adversarial_acc: float) -> float:
        """Calculate robustness score.
        
        Args:
            original_acc: Accuracy on original data.
            adversarial_acc: Accuracy on adversarial data.
            
        Returns:
            Robustness score (higher is better).
        """
        return adversarial_acc / original_acc if original_acc > 0 else 0.0
    
    @staticmethod
    def calculate_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            n_bins: Number of bins for calibration.
            
        Returns:
            Expected Calibration Error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def calculate_edge_case_performance(edge_case_predictions: np.ndarray, 
                                      normal_predictions: np.ndarray) -> float:
        """Calculate edge case performance score.
        
        Args:
            edge_case_predictions: Predictions on edge cases.
            normal_predictions: Predictions on normal cases.
            
        Returns:
            Edge case performance score.
        """
        # Measure consistency between edge case and normal predictions
        edge_case_consistency = len(np.unique(edge_case_predictions)) / len(np.unique(normal_predictions))
        return min(edge_case_consistency, 1.0)


class Leaderboard:
    """Leaderboard for tracking model performance across different tests."""
    
    def __init__(self):
        """Initialize leaderboard."""
        self.results = []
        self.metrics_calculator = MetricsCalculator()
    
    def add_result(self, model_name: str, test_results: Dict[str, Any], 
                   model_config: Optional[Dict[str, Any]] = None) -> None:
        """Add test results to leaderboard.
        
        Args:
            model_name: Name of the model.
            test_results: Results from behavioral tests.
            model_config: Optional model configuration.
        """
        # Extract metrics from test results
        accuracy = test_results.get("overall", {}).get("score", 0.0)
        robustness_score = test_results.get("robustness_test", {}).get("score", 0.0)
        calibration_error = 1.0 - test_results.get("calibration_test", {}).get("score", 0.0)
        edge_case_performance = test_results.get("edge_case_test", {}).get("score", 0.0)
        
        # Calculate overall score
        overall_score = np.mean([accuracy, robustness_score, 1.0 - calibration_error, edge_case_performance])
        
        result = {
            "model_name": model_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "accuracy": accuracy,
            "robustness_score": robustness_score,
            "calibration_error": calibration_error,
            "edge_case_performance": edge_case_performance,
            "overall_score": overall_score,
            "model_config": model_config or {},
            "test_results": test_results
        }
        
        self.results.append(result)
        logger.info(f"Added result for {model_name}: overall score = {overall_score:.3f}")
    
    def get_leaderboard(self, sort_by: str = "overall_score", ascending: bool = False) -> pd.DataFrame:
        """Get leaderboard as DataFrame.
        
        Args:
            sort_by: Column to sort by.
            ascending: Sort order.
            
        Returns:
            Leaderboard DataFrame.
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        return df.sort_values(sort_by, ascending=ascending)
    
    def get_top_models(self, n: int = 5, metric: str = "overall_score") -> pd.DataFrame:
        """Get top N models by specified metric.
        
        Args:
            n: Number of top models to return.
            metric: Metric to rank by.
            
        Returns:
            DataFrame with top N models.
        """
        leaderboard = self.get_leaderboard(sort_by=metric, ascending=False)
        return leaderboard.head(n)
    
    def get_model_comparison(self, model_names: List[str]) -> pd.DataFrame:
        """Compare specific models.
        
        Args:
            model_names: List of model names to compare.
            
        Returns:
            Comparison DataFrame.
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        return df[df["model_name"].isin(model_names)]
    
    def save_leaderboard(self, path: Union[str, Path]) -> None:
        """Save leaderboard to CSV file.
        
        Args:
            path: Path to save leaderboard.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.get_leaderboard()
        df.to_csv(path, index=False)
        logger.info(f"Leaderboard saved to {path}")
    
    def save_detailed_results(self, path: Union[str, Path]) -> None:
        """Save detailed results to JSON file.
        
        Args:
            path: Path to save detailed results.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {path}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all results.
        
        Returns:
            Summary statistics dictionary.
        """
        if not self.results:
            return {"message": "No results available"}
        
        df = pd.DataFrame(self.results)
        
        return {
            "total_models": len(df),
            "avg_overall_score": df["overall_score"].mean(),
            "best_model": df.loc[df["overall_score"].idxmax(), "model_name"],
            "best_score": df["overall_score"].max(),
            "avg_accuracy": df["accuracy"].mean(),
            "avg_robustness": df["robustness_score"].mean(),
            "avg_calibration_error": df["calibration_error"].mean(),
            "avg_edge_case_performance": df["edge_case_performance"].mean()
        }


class EvaluationSuite:
    """Comprehensive evaluation suite for behavioral testing."""
    
    def __init__(self):
        """Initialize evaluation suite."""
        self.leaderboard = Leaderboard()
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_model(self, model_name: str, model: Any, X_test: np.ndarray, 
                      y_test: np.ndarray, test_results: Dict[str, Any],
                      model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a model comprehensively.
        
        Args:
            model_name: Name of the model.
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            test_results: Results from behavioral tests.
            model_config: Optional model configuration.
            
        Returns:
            Comprehensive evaluation results.
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            y_prob = np.max(y_prob, axis=1)  # Max probability
        
        # Calculate metrics
        accuracy = self.metrics_calculator.calculate_accuracy(y_test, y_pred)
        
        # Extract other metrics from test results
        robustness_score = test_results.get("robustness_test", {}).get("score", 0.0)
        calibration_error = 1.0 - test_results.get("calibration_test", {}).get("score", 0.0)
        edge_case_performance = test_results.get("edge_case_test", {}).get("score", 0.0)
        
        # Calculate overall score
        overall_score = np.mean([accuracy, robustness_score, 1.0 - calibration_error, edge_case_performance])
        
        evaluation_result = {
            "model_name": model_name,
            "accuracy": accuracy,
            "robustness_score": robustness_score,
            "calibration_error": calibration_error,
            "edge_case_performance": edge_case_performance,
            "overall_score": overall_score,
            "model_config": model_config or {},
            "test_results": test_results
        }
        
        # Add to leaderboard
        self.leaderboard.add_result(model_name, test_results, model_config)
        
        return evaluation_result
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models.
        
        Args:
            model_names: List of model names to compare.
            
        Returns:
            Comparison results.
        """
        comparison_df = self.leaderboard.get_model_comparison(model_names)
        
        if comparison_df.empty:
            return {"message": "No results found for specified models"}
        
        return {
            "comparison_data": comparison_df.to_dict("records"),
            "summary": {
                "best_overall": comparison_df.loc[comparison_df["overall_score"].idxmax(), "model_name"],
                "best_accuracy": comparison_df.loc[comparison_df["accuracy"].idxmax(), "model_name"],
                "best_robustness": comparison_df.loc[comparison_df["robustness_score"].idxmax(), "model_name"],
                "best_calibration": comparison_df.loc[comparison_df["calibration_error"].idxmin(), "model_name"]
            }
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Returns:
            Evaluation report.
        """
        return {
            "leaderboard": self.leaderboard.get_leaderboard().to_dict("records"),
            "summary_stats": self.leaderboard.get_summary_stats(),
            "top_models": self.leaderboard.get_top_models().to_dict("records")
        }
