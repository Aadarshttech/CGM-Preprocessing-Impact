"""
step_statistics.py — Multi-run Statistical Significance Testing
===============================================================

Machine learning models (XGBoost, CNN, Transformer) have variance
due to random initialization and subsampling. A single run is not
enough to claim method A is better than method B.

This script:
1. Runs the prediction pipeline 5 times with different random seeds.
2. Collects RMSE across all runs.
3. Performs the Friedman test (non-parametric ANOVA) to check if 
   there are significant differences among methods.
4. If significant, performs post-hoc Nemenyi test to find exactly
   which methods are statistically different.
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import friedmanchisquare
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

from step4_features import prepare_column
from xgboost import XGBRegressor
from step6_CNN import run_cnn
import warnings
warnings.filterwarnings('ignore')


def run_multiple_seeds(df, cols, n_runs=5):
    """
    Run XGBoost, CNN, and Transformer multiple times with different random seeds.
    """
    results = []
    
    for run in range(n_runs):
        seed = 42 + run
        print(f"\n--- Run {run+1}/{n_runs} (Seed: {seed}) ---")
        
        for col in cols:
            print(f"  Evaluating {col}...")
            
            # XGBoost
            X_train, X_test, y_train, y_test = prepare_column(df, col)
            model = XGBRegressor(n_estimators=100, max_depth=6,
                                 learning_rate=0.05, random_state=seed, verbosity=0)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)
            xgb_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            
            # Record
            results.append({
                'run': run+1,
                'seed': seed,
                'preprocessing': col,
                'model': 'XGBoost',
                'rmse': xgb_rmse
            })
            # CNN (only 3 runs)
            if run < 3:
                cnn_rmse, _, _ = run_cnn(df, col, seed=seed)
                results.append({
                    'run': run+1, 'seed': seed, 'preprocessing': col,
                    'model': 'CNN', 'rmse': cnn_rmse
                })
                
                from step7_transformer import run_transformer
                tfm_rmse, _, _ = run_transformer(df, col, seed=seed)
                results.append({
                    'run': run+1, 'seed': seed, 'preprocessing': col,
                    'model': 'Transformer', 'rmse': tfm_rmse
                })

    return pd.DataFrame(results)


def statistical_analysis(results_df, save_path='results'):
    """
    Perform Friedman test and post-hoc Nemenyi test.
    """
    # 1. Summarize mean +/- std
    summary = results_df.groupby(['preprocessing', 'model'])['rmse'].agg(['mean', 'std']).reset_index()
    print("\n=== Mean +/- Std RMSE ===")
    print(summary.to_string(index=False))
    
    if sp is None:
        print("\n[!] scikit-posthocs not installed. Skipping Nemenyi test.")
        print("    pip install scikit-posthocs")
        return
        
    print("\n=== Statistical Tests ===")
    
    for model_name in ['XGBoost', 'CNN', 'Transformer']:
        print(f"\nModel: {model_name}")
        model_data = results_df[results_df['model'] == model_name]
        
        # Pivot table: rows=runs, cols=preprocessing, vals=rmse
        pivot = model_data.pivot(index='run', columns='preprocessing', values='rmse')
        
        # Friedman test
        stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        print(f"Friedman Test: Statistic={stat:.3f}, p-value={p:.3e}")
        
        if p < 0.05:
            print("  => Significant differences exist among preprocessing methods.")
            # Nemenyi post-hoc test
            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.columns = pivot.columns
            nemenyi.index = pivot.columns
            print("\nNemenyi p-values (p < 0.05 means significantly different):")
            print(nemenyi.round(3))
            nemenyi.to_csv(os.path.join(save_path, f'nemenyi_{model_name}.csv'))
        else:
            print("  => No significant differences detected.")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('data/hupa_smoothed.csv')
    
    cols = [c for c in df.columns if '_kalman' in c or '_smoothing_spline' in c or '_none' in c]
    
    print("Running Multi-Seed Statistical Analysis...")
    results_df = run_multiple_seeds(df, cols, n_runs=5)
    results_df.to_csv('results/multi_seed_results.csv', index=False)
    
    statistical_analysis(results_df)
