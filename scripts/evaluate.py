#!/usr/bin/env python3
"""Evaluation script for model behavioral testing framework."""

import argparse
import logging
from pathlib import Path
import yaml
import joblib
import numpy as np

from src import DataLoader, ModelFactory, BehavioralTester, EvaluationSuite, set_seed, setup_logging, Config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate models for behavioral testing")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    # Setup logging
    setup_logging(config.logging.level, config.logging.log_file)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    
    # Load model
    model = joblib.load(args.model_path)
    logger.info(f"Loaded model: {type(model).__name__}")
    
    # Load data
    data_loader = DataLoader(config.dataset.name, config.dataset.synthetic_config if config.dataset.name == "synthetic" else None)
    X_train, X_test, y_train, y_test = data_loader.load_data(
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state
    )
    
    logger.info(f"Loaded data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Run behavioral tests
    tester = BehavioralTester(model, config.model.config)
    test_results = tester.run_comprehensive_tests(X_test, y_test)
    
    logger.info("Behavioral testing completed")
    
    # Comprehensive evaluation
    evaluation_suite = EvaluationSuite()
    model_name = f"{config.dataset.name}_{config.model.type}"
    
    evaluation_result = evaluation_suite.evaluate_model(
        model_name=model_name,
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_results=test_results,
        model_config=config.model.config
    )
    
    logger.info(f"Evaluation completed for {model_name}")
    logger.info(f"Overall score: {evaluation_result['overall_score']:.4f}")
    
    # Save evaluation results
    eval_path = output_dir / f"{model_name}_evaluation.json"
    with open(eval_path, 'w') as f:
        yaml.dump(evaluation_result, f, default_flow_style=False)
    
    # Save leaderboard
    if config.output.save_leaderboard:
        leaderboard_path = output_dir / "leaderboard.csv"
        evaluation_suite.leaderboard.save_leaderboard(leaderboard_path)
        
        detailed_path = output_dir / "detailed_results.json"
        evaluation_suite.leaderboard.save_detailed_results(detailed_path)
    
    # Generate report
    report = evaluation_suite.generate_report()
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    
    logger.info(f"Evaluation results saved to {eval_path}")
    logger.info(f"Leaderboard saved to {leaderboard_path}")
    logger.info(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Overall Score: {evaluation_result['overall_score']:.4f}")
    print(f"Accuracy: {evaluation_result['accuracy']:.4f}")
    print(f"Robustness Score: {evaluation_result['robustness_score']:.4f}")
    print(f"Calibration Error: {evaluation_result['calibration_error']:.4f}")
    print(f"Edge Case Performance: {evaluation_result['edge_case_performance']:.4f}")
    print("="*50)
    
    return evaluation_result


if __name__ == "__main__":
    main()
