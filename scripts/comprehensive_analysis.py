#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Gold Price Prediction Project

This script provides a complete analysis and comparison of all validation approaches
and model performances implemented in the gold price prediction project.

Features:
- Loads and compares results from different validation methods
- Generates comprehensive performance reports
- Creates visualizations comparing model performances
- Provides insights and recommendations

Author: AI Assistant
Date: 2025-01-24
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalyzer:
    """Comprehensive analyzer for comparing all model validation results"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.analysis_results = {}
        self.comparison_data = []
        
    def load_all_results(self):
        """Load all available result files"""
        print("Loading all available results...")
        
        # Load optimized training results
        optimized_files = list(self.results_dir.glob("optimized_training_results_*.json"))
        if optimized_files:
            latest_optimized = max(optimized_files, key=os.path.getctime)
            with open(latest_optimized, 'r') as f:
                self.analysis_results['optimized_training'] = json.load(f)
            print(f"Loaded optimized training results: {latest_optimized.name}")
        
        # Load deep learning results
        dl_files = list(self.results_dir.glob("deep_learning_results_*.json"))
        if dl_files:
            latest_dl = max(dl_files, key=os.path.getctime)
            with open(latest_dl, 'r') as f:
                self.analysis_results['deep_learning'] = json.load(f)
            print(f"Loaded deep learning results: {latest_dl.name}")
        
        # Load walk-forward results
        wf_files = list(self.results_dir.glob("walk_forward_results_*.json"))
        if wf_files:
            latest_wf = max(wf_files, key=os.path.getctime)
            with open(latest_wf, 'r') as f:
                self.analysis_results['walk_forward'] = json.load(f)
            print(f"Loaded walk-forward results: {latest_wf.name}")
        
        # Load walk-forward summary
        wf_summary_files = list(self.results_dir.glob("walk_forward_summary_*.csv"))
        if wf_summary_files:
            latest_wf_summary = max(wf_summary_files, key=os.path.getctime)
            self.analysis_results['walk_forward_summary'] = pd.read_csv(latest_wf_summary)
            print(f"Loaded walk-forward summary: {latest_wf_summary.name}")
        
        print(f"Loaded {len(self.analysis_results)} result sets\n")
    
    def extract_comparison_data(self):
        """Extract comparable metrics from all validation approaches"""
        print("Extracting comparison data...")
        
        # Extract optimized training results
        if 'optimized_training' in self.analysis_results:
            opt_results = self.analysis_results['optimized_training']
            
            # Individual models
            for model_name, metrics in opt_results.get('model_results', {}).items():
                self.comparison_data.append({
                    'validation_method': 'Optimized Training (Time Series CV)',
                    'model_type': 'Individual',
                    'model_name': model_name,
                    'r2_score': metrics.get('r2', np.nan),
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'cv_score': metrics.get('cv_score', np.nan),
                    'cv_std': metrics.get('cv_std', np.nan)
                })
            
            # Ensemble models
            for model_name, metrics in opt_results.get('ensemble_results', {}).items():
                self.comparison_data.append({
                    'validation_method': 'Optimized Training (Time Series CV)',
                    'model_type': 'Ensemble',
                    'model_name': model_name,
                    'r2_score': metrics.get('r2', np.nan),
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'cv_score': metrics.get('cv_score', np.nan),
                    'cv_std': metrics.get('cv_std', np.nan)
                })
        
        # Extract deep learning results
        if 'deep_learning' in self.analysis_results:
            dl_results = self.analysis_results['deep_learning']
            
            for model_name, metrics in dl_results.get('model_results', {}).items():
                self.comparison_data.append({
                    'validation_method': 'Deep Learning (Time Series CV)',
                    'model_type': 'Deep Learning',
                    'model_name': model_name,
                    'r2_score': metrics.get('test_r2', np.nan),
                    'rmse': metrics.get('test_rmse', np.nan),
                    'mae': metrics.get('test_mae', np.nan),
                    'cv_score': metrics.get('cv_mean', np.nan),
                    'cv_std': metrics.get('cv_std', np.nan)
                })
        
        # Extract walk-forward results
        if 'walk_forward_summary' in self.analysis_results:
            wf_summary = self.analysis_results['walk_forward_summary']
            
            for _, row in wf_summary.iterrows():
                self.comparison_data.append({
                    'validation_method': 'Walk-Forward Analysis',
                    'model_type': 'Traditional ML',
                    'model_name': row['Model'],
                    'r2_score': row['Overall_R2'],
                    'rmse': row['Overall_RMSE'],
                    'mae': row['Overall_MAE'],
                    'cv_score': row['Mean_Fold_R2'],
                    'cv_std': row['Std_Fold_R2']
                })
        
        self.comparison_df = pd.DataFrame(self.comparison_data)
        print(f"Extracted {len(self.comparison_data)} model results for comparison\n")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("=" * 80)
        print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        if self.comparison_df.empty:
            print("No comparison data available.")
            return
        
        # Overall best performers
        print("\n1. OVERALL BEST PERFORMERS")
        print("-" * 40)
        
        # Best R² scores
        best_r2 = self.comparison_df.nlargest(5, 'r2_score')
        print("\nTop 5 Models by R² Score:")
        for idx, row in best_r2.iterrows():
            print(f"  {row['model_name']:15} ({row['validation_method']:25}): R² = {row['r2_score']:.4f}")
        
        # Best RMSE scores
        best_rmse = self.comparison_df.nsmallest(5, 'rmse')
        print("\nTop 5 Models by RMSE:")
        for idx, row in best_rmse.iterrows():
            print(f"  {row['model_name']:15} ({row['validation_method']:25}): RMSE = {row['rmse']:.2f}")
        
        # Performance by validation method
        print("\n\n2. PERFORMANCE BY VALIDATION METHOD")
        print("-" * 45)
        
        method_stats = self.comparison_df.groupby('validation_method').agg({
            'r2_score': ['mean', 'std', 'max'],
            'rmse': ['mean', 'std', 'min'],
            'mae': ['mean', 'std', 'min']
        }).round(4)
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns.values]
        
        print(method_stats)
        
        # Performance by model type
        print("\n\n3. PERFORMANCE BY MODEL TYPE")
        print("-" * 35)
        
        type_stats = self.comparison_df.groupby('model_type').agg({
            'r2_score': ['mean', 'std', 'max'],
            'rmse': ['mean', 'std', 'min'],
            'mae': ['mean', 'std', 'min']
        }).round(4)
        type_stats.columns = ['_'.join(col).strip() for col in type_stats.columns.values]
        
        print(type_stats)
        
        # Cross-validation stability
        print("\n\n4. CROSS-VALIDATION STABILITY ANALYSIS")
        print("-" * 45)
        
        # Filter out models with valid CV scores
        cv_data = self.comparison_df.dropna(subset=['cv_score', 'cv_std'])
        if not cv_data.empty:
            # Calculate stability score (higher is more stable)
            cv_data = cv_data.copy()
            cv_data['stability_score'] = -cv_data['cv_std'] / (abs(cv_data['cv_score']) + 1e-6)
            
            most_stable = cv_data.nlargest(5, 'stability_score')
            print("\nMost Stable Models (by CV std/mean ratio):")
            for idx, row in most_stable.iterrows():
                print(f"  {row['model_name']:15}: CV = {row['cv_score']:.4f} ± {row['cv_std']:.4f}")
        
        # Model-specific insights
        print("\n\n5. MODEL-SPECIFIC INSIGHTS")
        print("-" * 30)
        
        # Linear models performance
        linear_models = self.comparison_df[self.comparison_df['model_name'].isin(['lasso', 'ridge', 'elastic_net'])]
        if not linear_models.empty:
            print("\nLinear Models:")
            best_linear = linear_models.loc[linear_models['r2_score'].idxmax()]
            print(f"  Best: {best_linear['model_name']} (R² = {best_linear['r2_score']:.4f})")
            print(f"  Average R²: {linear_models['r2_score'].mean():.4f}")
        
        # Tree-based models performance
        tree_models = self.comparison_df[self.comparison_df['model_name'].isin(['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting'])]
        if not tree_models.empty:
            print("\nTree-based Models:")
            best_tree = tree_models.loc[tree_models['r2_score'].idxmax()]
            print(f"  Best: {best_tree['model_name']} (R² = {best_tree['r2_score']:.4f})")
            print(f"  Average R²: {tree_models['r2_score'].mean():.4f}")
        
        # Deep learning models performance
        dl_models = self.comparison_df[self.comparison_df['model_type'] == 'Deep Learning']
        if not dl_models.empty:
            print("\nDeep Learning Models:")
            best_dl = dl_models.loc[dl_models['r2_score'].idxmax()]
            print(f"  Best: {best_dl['model_name']} (R² = {best_dl['r2_score']:.4f})")
            print(f"  Average R²: {dl_models['r2_score'].mean():.4f}")
            print(f"  Note: Deep learning models showed poor performance, likely due to limited data or hyperparameter tuning")
        
        # Ensemble models performance
        ensemble_models = self.comparison_df[self.comparison_df['model_type'] == 'Ensemble']
        if not ensemble_models.empty:
            print("\nEnsemble Models:")
            best_ensemble = ensemble_models.loc[ensemble_models['r2_score'].idxmax()]
            print(f"  Best: {best_ensemble['model_name']} (R² = {best_ensemble['r2_score']:.4f})")
            print(f"  Average R²: {ensemble_models['r2_score'].mean():.4f}")
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        print("\n\n6. GENERATING VISUALIZATIONS")
        print("-" * 35)
        
        if self.comparison_df.empty:
            print("No data available for visualization.")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. R² Score Comparison
        plt.subplot(2, 3, 1)
        valid_r2 = self.comparison_df.dropna(subset=['r2_score'])
        if not valid_r2.empty:
            sns.boxplot(data=valid_r2, x='validation_method', y='r2_score')
            plt.title('R² Score by Validation Method')
            plt.xticks(rotation=45)
            plt.ylabel('R² Score')
        
        # 2. RMSE Comparison
        plt.subplot(2, 3, 2)
        valid_rmse = self.comparison_df.dropna(subset=['rmse'])
        if not valid_rmse.empty:
            sns.boxplot(data=valid_rmse, x='validation_method', y='rmse')
            plt.title('RMSE by Validation Method')
            plt.xticks(rotation=45)
            plt.ylabel('RMSE')
        
        # 3. Model Type Performance
        plt.subplot(2, 3, 3)
        valid_type = self.comparison_df.dropna(subset=['r2_score'])
        if not valid_type.empty:
            sns.boxplot(data=valid_type, x='model_type', y='r2_score')
            plt.title('R² Score by Model Type')
            plt.xticks(rotation=45)
            plt.ylabel('R² Score')
        
        # 4. Top 10 Models Bar Chart
        plt.subplot(2, 3, 4)
        top_models = self.comparison_df.nlargest(10, 'r2_score')
        if not top_models.empty:
            plt.barh(range(len(top_models)), top_models['r2_score'])
            plt.yticks(range(len(top_models)), [f"{row['model_name']}\n({row['validation_method'].split()[0]})" for _, row in top_models.iterrows()])
            plt.xlabel('R² Score')
            plt.title('Top 10 Models by R² Score')
            plt.gca().invert_yaxis()
        
        # 5. R² vs RMSE Scatter
        plt.subplot(2, 3, 5)
        valid_both = self.comparison_df.dropna(subset=['r2_score', 'rmse'])
        if not valid_both.empty:
            scatter = plt.scatter(valid_both['r2_score'], valid_both['rmse'], 
                                c=pd.Categorical(valid_both['model_type']).codes, 
                                alpha=0.7, s=60)
            plt.xlabel('R² Score')
            plt.ylabel('RMSE')
            plt.title('R² vs RMSE Trade-off')
            
            # Add legend for model types
            unique_types = valid_both['model_type'].unique()
            for i, model_type in enumerate(unique_types):
                plt.scatter([], [], c=f'C{i}', label=model_type)
            plt.legend()
        
        # 6. Cross-Validation Stability
        plt.subplot(2, 3, 6)
        cv_valid = self.comparison_df.dropna(subset=['cv_score', 'cv_std'])
        if not cv_valid.empty:
            plt.errorbar(range(len(cv_valid)), cv_valid['cv_score'], 
                        yerr=cv_valid['cv_std'], fmt='o', capsize=5)
            plt.xticks(range(len(cv_valid)), 
                      [f"{row['model_name']}\n({row['validation_method'].split()[0]})" for _, row in cv_valid.iterrows()], 
                      rotation=45)
            plt.ylabel('CV Score ± Std')
            plt.title('Cross-Validation Stability')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.results_dir / f"comprehensive_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to: {plot_file}")
        
        plt.show()
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        print("\n\n7. RECOMMENDATIONS AND INSIGHTS")
        print("-" * 40)
        
        if self.comparison_df.empty:
            print("No data available for recommendations.")
            return
        
        recommendations = []
        
        # Best overall model
        best_model = self.comparison_df.loc[self.comparison_df['r2_score'].idxmax()]
        recommendations.append(
            f"🏆 BEST OVERALL MODEL: {best_model['model_name']} "
            f"(R² = {best_model['r2_score']:.4f}, RMSE = {best_model['rmse']:.2f})"
        )
        
        # Validation method insights
        method_performance = self.comparison_df.groupby('validation_method')['r2_score'].mean().sort_values(ascending=False)
        best_method = method_performance.index[0]
        recommendations.append(
            f"📊 BEST VALIDATION METHOD: {best_method} "
            f"(Average R² = {method_performance.iloc[0]:.4f})"
        )
        
        # Model type insights
        type_performance = self.comparison_df.groupby('model_type')['r2_score'].mean().sort_values(ascending=False)
        best_type = type_performance.index[0]
        recommendations.append(
            f"🔧 BEST MODEL TYPE: {best_type} "
            f"(Average R² = {type_performance.iloc[0]:.4f})"
        )
        
        # Stability insights
        cv_data = self.comparison_df.dropna(subset=['cv_score', 'cv_std'])
        if not cv_data.empty:
            cv_data = cv_data.copy()
            cv_data['stability'] = cv_data['cv_std'] / (abs(cv_data['cv_score']) + 1e-6)
            most_stable = cv_data.loc[cv_data['stability'].idxmin()]
            recommendations.append(
                f"⚖️ MOST STABLE MODEL: {most_stable['model_name']} "
                f"(CV = {most_stable['cv_score']:.4f} ± {most_stable['cv_std']:.4f})"
            )
        
        # Deep learning insights
        dl_models = self.comparison_df[self.comparison_df['model_type'] == 'Deep Learning']
        if not dl_models.empty and dl_models['r2_score'].max() < 0:
            recommendations.append(
                "🧠 DEEP LEARNING MODELS: Showed poor performance. Consider:"
                "\n   - More data collection"
                "\n   - Better hyperparameter tuning"
                "\n   - Different architectures"
                "\n   - Feature engineering for sequential patterns"
            )
        
        # Linear model insights
        linear_models = self.comparison_df[self.comparison_df['model_name'].isin(['lasso', 'ridge', 'elastic_net'])]
        if not linear_models.empty and linear_models['r2_score'].max() > 0.9:
            recommendations.append(
                "📈 LINEAR MODELS: Excellent performance suggests:"
                "\n   - Strong linear relationships in the data"
                "\n   - Effective feature engineering"
                "\n   - Potential for simpler, interpretable models"
            )
        
        # General recommendations
        recommendations.extend([
            "\n🎯 GENERAL RECOMMENDATIONS:",
            "   • Use walk-forward analysis for realistic performance estimation",
            "   • Linear models (Lasso/Ridge) show excellent performance for this dataset",
            "   • Ensemble methods provide good balance of performance and stability",
            "   • Deep learning models need more tuning or data for this problem",
            "   • Time series cross-validation is crucial for temporal data"
        ])
        
        for rec in recommendations:
            print(rec)
    
    def _get_aggregated_stats(self, group_by_column):
        """Helper method to get aggregated statistics with proper column names"""
        stats = self.comparison_df.groupby(group_by_column).agg({
            'r2_score': ['mean', 'std', 'max'],
            'rmse': ['mean', 'std', 'min']
        }).round(4)
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        return stats.reset_index().to_dict('records')
    
    def save_comprehensive_report(self):
        """Save comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"comprehensive_analysis_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'analysis_summary': {
                'total_models_analyzed': len(self.comparison_df),
                'validation_methods': self.comparison_df['validation_method'].unique().tolist(),
                'model_types': self.comparison_df['model_type'].unique().tolist()
            },
            'best_performers': {
                'best_r2': self.comparison_df.loc[self.comparison_df['r2_score'].idxmax()].to_dict(),
                'best_rmse': self.comparison_df.loc[self.comparison_df['rmse'].idxmin()].to_dict()
            },
            'performance_by_method': self._get_aggregated_stats('validation_method'),
            'performance_by_type': self._get_aggregated_stats('model_type'),
            'detailed_results': self.comparison_df.to_dict('records')
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nComprehensive analysis report saved to: {report_file}")
        
        # Also save comparison data as CSV
        csv_file = self.results_dir / f"model_comparison_{timestamp}.csv"
        self.comparison_df.to_csv(csv_file, index=False)
        print(f"Model comparison data saved to: {csv_file}")
    
    def run_full_analysis(self, save_plots=True):
        """Run complete comprehensive analysis"""
        print("Starting Comprehensive Analysis...\n")
        
        self.load_all_results()
        self.extract_comparison_data()
        self.generate_performance_report()
        self.create_visualizations(save_plots)
        self.generate_recommendations()
        self.save_comprehensive_report()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Analysis of Gold Price Prediction Models')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ComprehensiveAnalyzer(args.results_dir)
    analyzer.run_full_analysis(save_plots=not args.no_plots)

if __name__ == "__main__":
    main()