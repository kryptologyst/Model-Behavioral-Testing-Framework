"""Streamlit demo application for model behavioral testing framework."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import yaml
import json

from src import DataLoader, ModelFactory, BehavioralTester, EvaluationSuite, set_seed, setup_logging, Config


# Page configuration
st.set_page_config(
    page_title="Model Behavioral Testing Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def create_model_comparison_chart(results_data: list) -> go.Figure:
    """Create comparison chart for multiple models."""
    if not results_data:
        return go.Figure()
    
    df = pd.DataFrame(results_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Score', 'Accuracy', 'Robustness', 'Calibration'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['overall_score', 'accuracy', 'robustness_score', 'calibration_error']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, pos in zip(metrics, positions):
        fig.add_trace(
            go.Bar(
                x=df['model_name'],
                y=df[metric],
                name=metric.replace('_', ' ').title(),
                showlegend=False
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        height=600,
        title_text="Model Performance Comparison",
        showlegend=False
    )
    
    return fig


def create_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
    """Create calibration plot."""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10
    )
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='red')
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='markers+lines',
        name='Model Calibration',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Calibration Plot',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧪 Model Behavioral Testing Framework</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ DISCLAIMER:</strong> This framework is for research and educational purposes only. 
        Results may be unstable or misleading. Not suitable for regulated decisions without human review.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Dataset selection
    dataset_options = ["iris", "wine", "breast_cancer", "synthetic"]
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options, index=0)
    
    # Model selection
    model_options = ModelFactory.get_available_models()
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)
    
    # Test parameters
    st.sidebar.subheader("Test Parameters")
    n_edge_cases = st.sidebar.slider("Number of Edge Cases", 5, 50, 10)
    epsilon = st.sidebar.slider("Adversarial Epsilon", 0.01, 0.5, 0.1, 0.01)
    n_samples = st.sidebar.slider("Robustness Test Samples", 10, 200, 50)
    
    # Random seed
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Quick Start", 
        "📊 Test Results", 
        "📈 Visualizations", 
        "🏆 Leaderboard", 
        "⚙️ Advanced"
    ])
    
    with tab1:
        st.header("Quick Start")
        
        if st.button("Run Behavioral Tests", type="primary"):
            with st.spinner("Running behavioral tests..."):
                # Set seed
                set_seed(seed)
                
                # Load data
                data_loader = DataLoader(selected_dataset)
                X_train, X_test, y_train, y_test = data_loader.load_data()
                
                # Create and train model
                model = ModelFactory.create_model(selected_model)
                model.fit(X_train, y_train)
                
                # Run tests
                tester = BehavioralTester(model)
                test_results = tester.run_comprehensive_tests(
                    X_test, y_test,
                    n_cases=n_edge_cases,
                    epsilon=epsilon,
                    n_samples=n_samples
                )
                
                # Store results in session state
                st.session_state.test_results = test_results
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.data_loader = data_loader
                
                st.success("Tests completed successfully!")
    
    with tab2:
        st.header("Test Results")
        
        if 'test_results' in st.session_state:
            results = st.session_state.test_results
            
            # Overall score
            overall_score = results['overall']['score']
            overall_passed = results['overall']['passed']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{overall_score:.3f}", 
                         delta="PASS" if overall_passed else "FAIL")
            with col2:
                st.metric("Tests Passed", f"{results['overall']['n_passed']}/{results['overall']['n_tests']}")
            with col3:
                st.metric("Model Type", selected_model)
            
            # Individual test results
            st.subheader("Individual Test Results")
            
            for test_name, test_result in results.items():
                if test_name == 'overall':
                    continue
                
                with st.expander(f"{test_name.replace('_', ' ').title()} - {'✅ PASSED' if test_result['passed'] else '❌ FAILED'}"):
                    st.write(f"**Score:** {test_result['score']:.3f}")
                    st.write("**Details:**")
                    for key, value in test_result['details'].items():
                        st.write(f"- {key}: {value}")
        
        else:
            st.info("Please run the behavioral tests first using the Quick Start tab.")
    
    with tab3:
        st.header("Visualizations")
        
        if 'test_results' in st.session_state and 'model' in st.session_state:
            results = st.session_state.test_results
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Test scores visualization
            st.subheader("Test Scores")
            test_names = [name for name in results.keys() if name != 'overall']
            test_scores = [results[name]['score'] for name in test_names]
            
            fig_scores = px.bar(
                x=test_names, 
                y=test_scores,
                title="Behavioral Test Scores",
                labels={'x': 'Test Name', 'y': 'Score'}
            )
            fig_scores.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Calibration plot
            if hasattr(model, 'predict_proba'):
                st.subheader("Calibration Analysis")
                y_prob = model.predict_proba(X_test)
                y_prob_max = np.max(y_prob, axis=1)
                
                cal_fig = create_calibration_plot(y_test, y_prob_max)
                st.plotly_chart(cal_fig, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_names = st.session_state.data_loader.feature_names
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        
        else:
            st.info("Please run the behavioral tests first using the Quick Start tab.")
    
    with tab4:
        st.header("Leaderboard")
        
        # Initialize evaluation suite
        if 'evaluation_suite' not in st.session_state:
            st.session_state.evaluation_suite = EvaluationSuite()
        
        evaluation_suite = st.session_state.evaluation_suite
        
        # Add current results to leaderboard
        if 'test_results' in st.session_state:
            model_name = f"{selected_dataset}_{selected_model}"
            
            if st.button("Add to Leaderboard"):
                evaluation_suite.leaderboard.add_result(
                    model_name, 
                    st.session_state.test_results,
                    {"dataset": selected_dataset, "model": selected_model}
                )
                st.success(f"Added {model_name} to leaderboard!")
        
        # Display leaderboard
        leaderboard_df = evaluation_suite.leaderboard.get_leaderboard()
        
        if not leaderboard_df.empty:
            st.subheader("Current Leaderboard")
            st.dataframe(leaderboard_df, use_container_width=True)
            
            # Top models chart
            top_models = evaluation_suite.leaderboard.get_top_models(5)
            if not top_models.empty:
                comparison_fig = create_model_comparison_chart(top_models.to_dict('records'))
                st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("No models in leaderboard yet. Run tests and add results to see the leaderboard.")
    
    with tab5:
        st.header("Advanced Configuration")
        
        st.subheader("Custom Configuration")
        
        # Configuration editor
        default_config = {
            "dataset": {
                "name": selected_dataset,
                "test_size": 0.3,
                "random_state": seed
            },
            "model": {
                "type": selected_model,
                "config": ModelFactory.get_default_config(selected_model)
            },
            "testing": {
                "edge_case_test": {"n_cases": n_edge_cases},
                "robustness_test": {"epsilon": epsilon, "n_samples": n_samples},
                "calibration_test": {"n_bins": 10}
            }
        }
        
        config_text = st.text_area(
            "Configuration YAML",
            value=yaml.dump(default_config, default_flow_style=False),
            height=300
        )
        
        if st.button("Load Custom Configuration"):
            try:
                custom_config = yaml.safe_load(config_text)
                st.session_state.custom_config = Config(custom_config)
                st.success("Custom configuration loaded successfully!")
            except Exception as e:
                st.error(f"Error loading configuration: {e}")
        
        # Export results
        if 'test_results' in st.session_state:
            st.subheader("Export Results")
            
            if st.button("Export Results as JSON"):
                results_json = json.dumps(st.session_state.test_results, indent=2)
                st.download_button(
                    label="Download Results",
                    data=results_json,
                    file_name=f"behavioral_test_results_{selected_dataset}_{selected_model}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
