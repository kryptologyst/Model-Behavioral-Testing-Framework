#!/usr/bin/env python3
"""Example script demonstrating the Model Behavioral Testing Framework."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src import DataLoader, ModelFactory, BehavioralTester, set_seed, setup_logging


def main():
    """Main example function."""
    print("🧪 Model Behavioral Testing Framework - Example")
    print("=" * 60)
    
    # Setup
    set_seed(42)
    setup_logging("INFO")
    
    # Load data
    print("\n1. Loading Iris dataset...")
    data_loader = DataLoader("iris")
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    # Train model
    print("\n2. Training Random Forest model...")
    model = ModelFactory.create_model("random_forest")
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Test accuracy: {test_score:.4f}")
    
    # Run behavioral tests
    print("\n3. Running behavioral tests...")
    tester = BehavioralTester(model)
    test_results = tester.run_comprehensive_tests(X_test, y_test)
    
    print(f"   Overall score: {test_results['overall']['score']:.4f}")
    print(f"   Tests passed: {test_results['overall']['n_passed']}/{test_results['overall']['n_tests']}")
    
    # Display detailed results
    print("\n4. Detailed test results:")
    for test_name, result in test_results.items():
        if test_name == 'overall':
            continue
        
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status} (score: {result['score']:.3f})")
    
    # Create visualization
    print("\n5. Creating visualization...")
    test_names = [name for name in test_results.keys() if name != 'overall']
    test_scores = [test_results[name]['score'] for name in test_names]
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if score > 0.7 else 'orange' if score > 0.5 else 'red' for score in test_scores]
    bars = plt.bar(test_names, test_scores, color=colors)
    
    plt.title('Behavioral Test Scores', fontsize=16)
    plt.xlabel('Test Name', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add score labels
    for bar, score in zip(bars, test_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    plt.savefig(assets_dir / "behavioral_test_scores.png", dpi=300, bbox_inches='tight')
    print(f"   Plot saved to: {assets_dir / 'behavioral_test_scores.png'}")
    
    # Save results
    print("\n6. Saving results...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_path = results_dir / "iris_random_forest_results.json"
    tester.save_results(results_path)
    print(f"   Results saved to: {results_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: Random Forest on Iris dataset")
    print(f"Overall Behavioral Score: {test_results['overall']['score']:.4f}")
    print(f"Status: {'✅ PASSED' if test_results['overall']['passed'] else '❌ FAILED'}")
    print("\n⚠️  DISCLAIMER: Results are for research/educational purposes only.")
    print("   Not suitable for regulated decisions without human review.")
    print("=" * 60)


if __name__ == "__main__":
    main()
