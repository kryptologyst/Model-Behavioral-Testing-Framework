#!/usr/bin/env python3
"""Training script for model behavioral testing framework."""

import argparse
import logging
from pathlib import Path
import yaml
import joblib
import numpy as np

from src import DataLoader, ModelFactory, BehavioralTester, set_seed, setup_logging, Config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a model for behavioral testing")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Output directory for trained models")
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
    
    logger.info("Starting model training")
    logger.info(f"Configuration: {config.dataset.name} dataset, {config.model.type} model")
    
    # Load data
    data_loader = DataLoader(config.dataset.name, config.dataset.synthetic_config if config.dataset.name == "synthetic" else None)
    X_train, X_test, y_train, y_test = data_loader.load_data(
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state
    )
    
    logger.info(f"Loaded data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Create and train model
    model = ModelFactory.create_model(config.model.type, config.model.config)
    model.fit(X_train, y_train)
    
    logger.info(f"Trained {config.model.type} model")
    
    # Evaluate on test set
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Run behavioral tests
    tester = BehavioralTester(model, config.model.config)
    test_results = tester.run_comprehensive_tests(X_test, y_test)
    
    logger.info("Behavioral testing completed")
    logger.info(f"Overall test score: {test_results['overall']['score']:.4f}")
    
    # Save model and results
    model_path = output_dir / f"{config.dataset.name}_{config.model.type}_model.pkl"
    joblib.dump(model, model_path)
    
    results_path = output_dir / f"{config.dataset.name}_{config.model.type}_results.json"
    tester.save_results(results_path)
    
    # Save metadata
    metadata = {
        "dataset": config.dataset.name,
        "model_type": config.model.type,
        "model_config": config.model.config,
        "train_score": train_score,
        "test_score": test_score,
        "test_results": test_results,
        "data_metadata": data_loader.get_metadata()
    }
    
    metadata_path = output_dir / f"{config.dataset.name}_{config.model.type}_metadata.json"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model, test_results


if __name__ == "__main__":
    main()
