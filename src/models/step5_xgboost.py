import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
from step4_features import prepare_column

# ─────────────────────────────────────────────────
# CONCEPT — XGBoost:
# Builds an ensemble of decision trees sequentially.
# Each tree corrects the errors of the previous one.
# Takes flat feature vector (12 glucose values) as input.
# Does NOT understand time order — just sees 12 numbers.
# ─────────────────────────────────────────────────

def run_xgboost(df, col_name):
    X_train, X_test, y_train, y_test = prepare_column(df, col_name)
    
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse, y_test, y_pred


if __name__ == '__main__':
    df = pd.read_csv('data/hupa_smoothed.csv')
    os.makedirs('results', exist_ok=True)
    
    # All 8 preprocessed columns
    cols = [c for c in df.columns 
            if '_kalman' in c or '_smoothing_spline' in c or '_none' in c]
    
    print("XGBoost Results:")
    print("-" * 45)
    
    results = {}
    for col in cols:
        rmse, y_test, y_pred = run_xgboost(df, col)
        results[col] = rmse
        print(f"  {col:<35} RMSE: {rmse:.4f} mg/dL")
    
    # Plot best result
    best_col = min(results, key=results.get)
    _, y_test, y_pred = run_xgboost(df, best_col)
    
    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:300], 'k-', label='True', linewidth=1.5)
    plt.plot(y_pred[:300], 'b--', label='XGBoost', linewidth=1.5)
    plt.title(f'XGBoost Best Result — {best_col} '
              f'(RMSE: {results[best_col]:.2f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/xgboost_best.png', dpi=150)
    
    # Save results
    pd.DataFrame(list(results.items()), 
                 columns=['preprocessing', 'xgboost_rmse']
                 ).to_csv('results/xgboost_results.csv', index=False)
    print("\nSaved: results/xgboost_results.csv")