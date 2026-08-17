import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

class WalkForwardAnalyzer:
    """
    Walk-forward analysis for time series model validation.
    
    This class implements walk-forward analysis, which is considered the gold standard
    for time series model validation as it simulates real-world trading conditions.
    """
    
    def __init__(self, 
                 initial_train_size: int = 252,  # ~1 year of trading days
                 step_size: int = 21,             # ~1 month
                 prediction_horizon: int = 1,     # days ahead
                 expanding_window: bool = True):   # expanding vs rolling window
        """
        Initialize walk-forward analyzer.
        
        Args:
            initial_train_size: Initial training window size
            step_size: Number of periods to step forward
            prediction_horizon: Number of periods ahead to predict
            expanding_window: If True, use expanding window; if False, use rolling window
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.prediction_horizon = prediction_horizon
        self.expanding_window = expanding_window
        self.results = []
        
    def generate_walk_forward_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        splits = []
        n_samples = len(X)
        
        # Start from initial_train_size and step forward
        for i in range(self.initial_train_size, n_samples - self.prediction_horizon, self.step_size):
            # Define training window
            if self.expanding_window:
                train_start = 0
            else:
                train_start = max(0, i - self.initial_train_size)
            
            train_end = i
            test_start = i
            test_end = min(i + self.step_size, n_samples - self.prediction_horizon)
            
            # Extract splits
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start + self.prediction_horizon - 1:test_end + self.prediction_horizon - 1]
            
            # Ensure we have valid data
            if len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0:
                splits.append((X_train, X_test, y_train, y_test))
        
        return splits
    
    def evaluate_model(self, 
                      model_func: Callable,
                      X: np.ndarray, 
                      y: np.ndarray,
                      model_params: Optional[Dict[str, Any]] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Perform walk-forward analysis on a single model.
        
        Args:
            model_func: Function that returns a fitted model (e.g., lambda: RandomForestRegressor())
            X: Feature matrix
            y: Target vector
            model_params: Parameters to pass to model_func
            model_name: Name of the model for tracking
            
        Returns:
            Dictionary with walk-forward results
        """
        if model_params is None:
            model_params = {}
        
        splits = self.generate_walk_forward_splits(X, y)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        print(f"Performing walk-forward analysis for {model_name} with {len(splits)} folds...")
        
        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
            try:
                # Create and train model
                model = model_func(**model_params)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics for this fold
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                fold_result = {
                    'fold': fold_idx,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                }
                
                fold_results.append(fold_result)
                all_predictions.extend(y_pred)
                all_actuals.extend(y_test)
                
                if fold_idx % 5 == 0:  # Print progress every 5 folds
                    print(f"  Fold {fold_idx + 1}/{len(splits)} - R²: {r2:.4f}, RMSE: {rmse:.2f}")
                    
            except Exception as e:
                print(f"  Warning: Fold {fold_idx} failed with error: {str(e)}")
                continue
        
        # Calculate overall metrics
        if len(all_predictions) > 0:
            overall_mse = mean_squared_error(all_actuals, all_predictions)
            overall_mae = mean_absolute_error(all_actuals, all_predictions)
            overall_r2 = r2_score(all_actuals, all_predictions)
            overall_rmse = np.sqrt(overall_mse)
        else:
            overall_mse = overall_mae = overall_r2 = overall_rmse = np.nan
        
        # Calculate fold statistics
        fold_metrics = pd.DataFrame(fold_results)
        
        if len(fold_metrics) > 0:
            fold_stats = {
                'mean_r2': fold_metrics['r2'].mean(),
                'std_r2': fold_metrics['r2'].std(),
                'mean_rmse': fold_metrics['rmse'].mean(),
                'std_rmse': fold_metrics['rmse'].std(),
                'mean_mae': fold_metrics['mae'].mean(),
                'std_mae': fold_metrics['mae'].std(),
                'successful_folds': len(fold_metrics),
                'total_folds': len(splits)
            }
        else:
            fold_stats = {
                'mean_r2': np.nan,
                'std_r2': np.nan,
                'mean_rmse': np.nan,
                'std_rmse': np.nan,
                'mean_mae': np.nan,
                'std_mae': np.nan,
                'successful_folds': 0,
                'total_folds': len(splits)
            }
        
        result = {
            'model_name': model_name,
            'overall_metrics': {
                'mse': overall_mse,
                'mae': overall_mae,
                'r2': overall_r2,
                'rmse': overall_rmse
            },
            'fold_statistics': fold_stats,
            'fold_results': fold_results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'parameters': {
                'initial_train_size': self.initial_train_size,
                'step_size': self.step_size,
                'prediction_horizon': self.prediction_horizon,
                'expanding_window': self.expanding_window
            }
        }
        
        self.results.append(result)
        return result
    
    def evaluate_multiple_models(self, 
                               models_config: Dict[str, Dict[str, Any]],
                               X: np.ndarray, 
                               y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Perform walk-forward analysis on multiple models.
        
        Args:
            models_config: Dictionary with model configurations
                          Format: {'model_name': {'func': model_function, 'params': model_params}}
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        print(f"Starting walk-forward analysis for {len(models_config)} models...")
        print(f"Configuration: {self.initial_train_size} initial train size, {self.step_size} step size")
        print(f"Window type: {'Expanding' if self.expanding_window else 'Rolling'}")
        print("=" * 60)
        
        for model_name, config in models_config.items():
            print(f"\nEvaluating {model_name}...")
            
            model_func = config['func']
            model_params = config.get('params', {})
            
            result = self.evaluate_model(
                model_func=model_func,
                X=X,
                y=y,
                model_params=model_params,
                model_name=model_name
            )
            
            results[model_name] = result
            
            # Print summary for this model
            print(f"  Overall R²: {result['overall_metrics']['r2']:.4f}")
            print(f"  Overall RMSE: {result['overall_metrics']['rmse']:.2f}")
            print(f"  Mean fold R²: {result['fold_statistics']['mean_r2']:.4f} (±{result['fold_statistics']['std_r2']:.4f})")
            print(f"  Successful folds: {result['fold_statistics']['successful_folds']}/{result['fold_statistics']['total_folds']}")
        
        return results
    
    def get_best_model(self, results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on overall R² score.
        
        Args:
            results: Results dictionary from evaluate_multiple_models
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        valid_results = {name: result for name, result in results.items() 
                        if not np.isnan(result['overall_metrics']['r2'])}
        
        if not valid_results:
            raise ValueError("No valid results found")
        
        best_model = max(valid_results.items(), key=lambda x: x[1]['overall_metrics']['r2'])
        return best_model[0], best_model[1]
    
    def generate_performance_summary(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate a summary DataFrame of model performance.
        
        Args:
            results: Results dictionary from evaluate_multiple_models
            
        Returns:
            DataFrame with performance summary
        """
        summary_data = []
        
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Overall_R2': result['overall_metrics']['r2'],
                'Overall_RMSE': result['overall_metrics']['rmse'],
                'Overall_MAE': result['overall_metrics']['mae'],
                'Mean_Fold_R2': result['fold_statistics']['mean_r2'],
                'Std_Fold_R2': result['fold_statistics']['std_r2'],
                'Mean_Fold_RMSE': result['fold_statistics']['mean_rmse'],
                'Std_Fold_RMSE': result['fold_statistics']['std_rmse'],
                'Successful_Folds': result['fold_statistics']['successful_folds'],
                'Total_Folds': result['fold_statistics']['total_folds'],
                'Success_Rate': result['fold_statistics']['successful_folds'] / result['fold_statistics']['total_folds']
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('Overall_R2', ascending=False)
    
    def plot_walk_forward_results(self, results: Dict[str, Dict[str, Any]], metric: str = 'r2'):
        """
        Plot walk-forward analysis results over time.
        
        Args:
            results: Results dictionary from evaluate_multiple_models
            metric: Metric to plot ('r2', 'rmse', 'mae')
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
            if len(results) == 1:
                axes = [axes]
            
            for idx, (model_name, result) in enumerate(results.items()):
                fold_results = pd.DataFrame(result['fold_results'])
                
                if len(fold_results) > 0:
                    axes[idx].plot(fold_results['fold'], fold_results[metric], marker='o', linewidth=2)
                    axes[idx].set_title(f'{model_name} - {metric.upper()} over time')
                    axes[idx].set_xlabel('Fold')
                    axes[idx].set_ylabel(metric.upper())
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add horizontal line for mean
                    mean_val = fold_results[metric].mean()
                    axes[idx].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                                    label=f'Mean: {mean_val:.4f}')
                    axes[idx].legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot results.")
    
    def save_results(self, results: Dict[str, Dict[str, Any]], filepath: str):
        """
        Save walk-forward analysis results to file.
        
        Args:
            results: Results dictionary
            filepath: Path to save results
        """
        import json
        import numpy as np
        
        def convert_numpy_to_list(obj):
            """Recursively convert numpy arrays to lists for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_list(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_numpy_to_list(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")