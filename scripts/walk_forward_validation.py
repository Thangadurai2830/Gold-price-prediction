#!/usr/bin/env python3
"""
Walk-Forward Analysis Script for Gold Price Prediction

This script performs walk-forward analysis on various models to simulate
real-world trading conditions and provide robust model validation.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collector import GoldDataCollector
from ml.robust_feature_engineering import RobustFeatureEngineer
from ml.walk_forward_analysis import WalkForwardAnalyzer
from ml.ensemble_methods import EnsembleManager

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Perform walk-forward analysis for gold price prediction')
    parser.add_argument('--real-data', action='store_true', help='Use real data instead of synthetic')
    parser.add_argument('--initial-train-size', type=int, default=252, help='Initial training window size (days)')
    parser.add_argument('--step-size', type=int, default=21, help='Step size for walk-forward (days)')
    parser.add_argument('--prediction-horizon', type=int, default=1, help='Prediction horizon (days ahead)')
    parser.add_argument('--expanding-window', action='store_true', default=True, help='Use expanding window (vs rolling)')
    parser.add_argument('--rolling-window', action='store_true', help='Use rolling window (vs expanding)')
    parser.add_argument('--include-ensembles', action='store_true', help='Include ensemble methods in analysis')
    return parser.parse_args()

def create_model_configurations():
    """
    Create model configurations for walk-forward analysis.
    
    Returns:
        Dictionary with model configurations
    """
    models_config = {
        'random_forest': {
            'func': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'params': {}
        },
        'gradient_boosting': {
            'func': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'params': {}
        },
        'xgboost': {
            'func': lambda: xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
            'params': {}
        },
        'lightgbm': {
            'func': lambda: lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
            'params': {}
        },
        'ridge': {
            'func': lambda: Ridge(alpha=1.0, random_state=42),
            'params': {}
        },
        'lasso': {
            'func': lambda: Lasso(alpha=1.0, random_state=42, max_iter=1000),
            'params': {}
        },
        'elastic_net': {
            'func': lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=1000),
            'params': {}
        },
        'svr': {
            'func': lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
            'params': {}
        }
    }
    
    return models_config

def prepare_data_for_walk_forward(df: pd.DataFrame, feature_engineer: RobustFeatureEngineer, 
                                 prediction_horizon: int = 1):
    """
    Prepare data for walk-forward analysis.
    
    Args:
        df: Input dataframe
        feature_engineer: Feature engineering instance
        prediction_horizon: Number of steps ahead to predict
        
    Returns:
        Tuple of (X, y) prepared for walk-forward analysis
    """
    print("Preparing features for walk-forward analysis...")
    
    # Prepare training data with proper temporal constraints
    X, y = feature_engineer.prepare_training_data(
        df, 
        price_col='close',
        prediction_horizon=prediction_horizon
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def run_ensemble_walk_forward(wf_analyzer: WalkForwardAnalyzer, X: np.ndarray, y: np.ndarray):
    """
    Run walk-forward analysis on ensemble methods.
    
    Args:
        wf_analyzer: Walk-forward analyzer instance
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary with ensemble results
    """
    print("\nRunning walk-forward analysis on ensemble methods...")
    
    # Create base models for ensembles
    base_models = {
        'rf': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        'gb': GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
        'ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    ensemble_results = {}
    splits = wf_analyzer.generate_walk_forward_splits(X, y)
    
    print(f"Evaluating ensembles with {len(splits)} folds...")
    
    all_predictions = {'stacking': [], 'blending': []}
    all_actuals = []
    
    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
        try:
            # Initialize ensemble manager
            ensemble_manager = EnsembleManager(base_models)
            
            # Train ensemble models
            ensemble_manager.train_stacking(X_train, y_train)
            ensemble_manager.train_blending(X_train, y_train)
            
            # Make predictions
            stacking_pred = ensemble_manager.predict_stacking(X_test)
            blending_pred = ensemble_manager.predict_blending(X_test)
            
            all_predictions['stacking'].extend(stacking_pred)
            all_predictions['blending'].extend(blending_pred)
            
            if fold_idx == 0:  # Only add actuals once
                all_actuals.extend(y_test)
            
            if fold_idx % 5 == 0:
                print(f"  Ensemble fold {fold_idx + 1}/{len(splits)} completed")
                
        except Exception as e:
            print(f"  Warning: Ensemble fold {fold_idx} failed with error: {str(e)}")
            continue
    
    # Calculate ensemble metrics
    for ensemble_name, predictions in all_predictions.items():
        if len(predictions) > 0 and len(all_actuals) > 0:
            # Ensure same length
            min_len = min(len(predictions), len(all_actuals))
            pred_subset = predictions[:min_len]
            actual_subset = all_actuals[:min_len]
            
            mse = np.mean((np.array(actual_subset) - np.array(pred_subset)) ** 2)
            mae = np.mean(np.abs(np.array(actual_subset) - np.array(pred_subset)))
            
            # Calculate R² manually
            ss_res = np.sum((np.array(actual_subset) - np.array(pred_subset)) ** 2)
            ss_tot = np.sum((np.array(actual_subset) - np.mean(actual_subset)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            rmse = np.sqrt(mse)
            
            ensemble_results[f'ensemble_{ensemble_name}'] = {
                'model_name': f'ensemble_{ensemble_name}',
                'overall_metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                },
                'fold_statistics': {
                    'mean_r2': r2,
                    'std_r2': 0.0,
                    'mean_rmse': rmse,
                    'std_rmse': 0.0,
                    'mean_mae': mae,
                    'std_mae': 0.0,
                    'successful_folds': len(splits),
                    'total_folds': len(splits)
                },
                'predictions': pred_subset,
                'actuals': actual_subset
            }
    
    return ensemble_results

def main():
    """Main walk-forward analysis function."""
    args = parse_arguments()
    
    # Handle window type
    expanding_window = not args.rolling_window if args.rolling_window else args.expanding_window
    
    print("=" * 70)
    print("WALK-FORWARD ANALYSIS FOR GOLD PRICE PREDICTION")
    print("=" * 70)
    print(f"Initial training size: {args.initial_train_size} days")
    print(f"Step size: {args.step_size} days")
    print(f"Prediction horizon: {args.prediction_horizon} days")
    print(f"Window type: {'Expanding' if expanding_window else 'Rolling'}")
    print(f"Include ensembles: {args.include_ensembles}")
    print(f"Using real data: {args.real_data}")
    
    # Initialize data collector and feature engineer
    collector = GoldDataCollector()
    feature_engineer = RobustFeatureEngineer()
    
    # Load data
    if args.real_data:
        print("\nFetching real gold price data...")
        df = collector.fetch_gold_prices_yahoo()
        print(f"Loaded {len(df)} days of real data")
    else:
        print("\nGenerating synthetic data...")
        df = collector.generate_synthetic_data(days=1500)  # More data for walk-forward
        print(f"Generated {len(df)} days of synthetic data")
    
    # Prepare data
    X, y = prepare_data_for_walk_forward(df, feature_engineer, args.prediction_horizon)
    
    # Initialize walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer(
        initial_train_size=args.initial_train_size,
        step_size=args.step_size,
        prediction_horizon=args.prediction_horizon,
        expanding_window=expanding_window
    )
    
    # Create model configurations
    models_config = create_model_configurations()
    
    print("\n" + "=" * 70)
    print("STARTING WALK-FORWARD ANALYSIS")
    print("=" * 70)
    
    # Run walk-forward analysis on individual models
    results = wf_analyzer.evaluate_multiple_models(models_config, X, y)
    
    # Run ensemble analysis if requested
    if args.include_ensembles:
        print("\n" + "=" * 50)
        print("ENSEMBLE WALK-FORWARD ANALYSIS")
        print("=" * 50)
        
        try:
            ensemble_results = run_ensemble_walk_forward(wf_analyzer, X, y)
            results.update(ensemble_results)
        except Exception as e:
            print(f"Warning: Ensemble analysis failed with error: {str(e)}")
    
    # Generate performance summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 70)
    
    summary_df = wf_analyzer.generate_performance_summary(results)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Find best model
    try:
        best_model_name, best_model_results = wf_analyzer.get_best_model(results)
        print(f"\n" + "=" * 50)
        print("BEST MODEL SUMMARY")
        print("=" * 50)
        print(f"Best Model: {best_model_name}")
        print(f"Overall R²: {best_model_results['overall_metrics']['r2']:.4f}")
        print(f"Overall RMSE: {best_model_results['overall_metrics']['rmse']:.2f}")
        print(f"Mean Fold R²: {best_model_results['fold_statistics']['mean_r2']:.4f} (±{best_model_results['fold_statistics']['std_r2']:.4f})")
        print(f"Success Rate: {best_model_results['fold_statistics']['successful_folds']}/{best_model_results['fold_statistics']['total_folds']}")
    except ValueError as e:
        print(f"\nWarning: Could not determine best model: {str(e)}")
        best_model_name = "none"
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare final results
    final_results = {
        'timestamp': timestamp,
        'parameters': {
            'initial_train_size': args.initial_train_size,
            'step_size': args.step_size,
            'prediction_horizon': args.prediction_horizon,
            'expanding_window': expanding_window,
            'include_ensembles': args.include_ensembles,
            'real_data': args.real_data
        },
        'walk_forward_results': results,
        'performance_summary': summary_df.to_dict('records'),
        'best_model': {
            'name': best_model_name,
            'results': best_model_results if best_model_name != "none" else None
        }
    }
    
    # Save results
    results_file = os.path.join(results_dir, f"walk_forward_results_{timestamp}.json")
    wf_analyzer.save_results(final_results, results_file)
    
    # Save summary CSV
    summary_file = os.path.join(results_dir, f"walk_forward_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Print key insights
    print("\nKey Insights:")
    print(f"• Analyzed {len(results)} models with walk-forward validation")
    print(f"• Best performing model: {best_model_name}")
    if best_model_name != "none":
        print(f"• Best model R²: {best_model_results['overall_metrics']['r2']:.4f}")
    print(f"• Analysis simulated {wf_analyzer.step_size}-day trading periods")
    print(f"• Used {'expanding' if expanding_window else 'rolling'} window validation")

if __name__ == "__main__":
    main()