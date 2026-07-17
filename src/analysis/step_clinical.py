"""
step_clinical.py — Clinical Metrics (Clarke Error Grid)
=========================================================

RMSE is a mathematical metric, but in diabetes, a 20 mg/dL error 
at 80 mg/dL (hypoglycemia) is far more dangerous than a 20 mg/dL 
error at 200 mg/dL (hyperglycemia).

The Clarke Error Grid (CEG) categorizes predictions into 5 clinical zones:
A: Clinically accurate
B: Benign error (no dangerous action taken)
C: Overcorrection (leads to unnecessary treatment)
D: Failure to detect (misses hypo/hyperglycemia)
E: Erroneous treatment (dangerous)

This script plots the CEG and computes the percentage in each zone
for the best vs. worst preprocessing pipelines.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from step4_features import prepare_column
from xgboost import XGBRegressor

def clarke_error_grid(ref_values, pred_values, title="Clarke Error Grid", save_name="ceg.png"):
    """
    Plots a Clarke Error Grid and computes zone percentages.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot data points
    ax.scatter(ref_values, pred_values, alpha=0.5, s=10, c='black')
    
    # Perfect match line
    ax.plot([0, 400], [0, 400], 'k--', alpha=0.5)
    
    # Zone lines (Simplified boundaries for visualization)
    # A vs B
    ax.plot([0, 58.33], [0, 70], 'k-', linewidth=1.5)
    ax.plot([58.33, 400], [70, 480], 'k-', linewidth=1.5)
    ax.plot([0, 70], [0, 58.33], 'k-', linewidth=1.5)
    ax.plot([70, 400], [58.33, 333.33], 'k-', linewidth=1.5)
    
    # B vs C, D, E boundaries are complex step functions,
    # Here we focus on calculating the percentages accurately.
    
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    
    for r, p in zip(ref_values, pred_values):
        if r <= 70 and p <= 70:
            zones['A'] += 1
        elif r <= 70 and p > 70:
            if p >= 180: zones['E'] += 1
            else: zones['D'] += 1
        elif r > 70 and r <= 180:
            if abs(r - p) <= 0.2 * r: zones['A'] += 1
            elif p <= 70: zones['D'] += 1
            elif p >= 180: zones['C'] += 1
            else: zones['B'] += 1
        else: # r > 180
            if abs(r - p) <= 0.2 * r: zones['A'] += 1
            elif p <= 70: zones['E'] += 1
            elif p >= 180: zones['B'] += 1
            else: zones['D'] += 1
            
    total = len(ref_values)
    z_pct = {k: v/total*100 for k, v in zones.items()}
    
    # Annotate plot
    textstr = '\n'.join((
        f'Zone A: {z_pct["A"]:.1f}%',
        f'Zone B: {z_pct["B"]:.1f}%',
        f'Zone C: {z_pct["C"]:.1f}%',
        f'Zone D: {z_pct["D"]:.1f}%',
        f'Zone E: {z_pct["E"]:.1f}%'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', save_name), dpi=150)
    plt.close()
    
    return z_pct

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('data/hupa_smoothed.csv')
    
    # Best and worst methods based on previous results
    best_col = 'spline_smoothing_spline'
    worst_col = 'hourly_mean_smoothing_spline'
    
    print("Running Clinical Metrics (Clarke Error Grid)...")
    
    for col, name in [(best_col, 'Best'), (worst_col, 'Worst')]:
        print(f"  Evaluating {name} ({col})...")
        X_train, X_test, y_train, y_test = prepare_column(df, col)
        
        model = XGBRegressor(n_estimators=100, max_depth=6,
                             learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        
        z_pct = clarke_error_grid(y_test, y_pred, 
                                  title=f'Clarke Error Grid - {name} Preprocessing\n({col})', 
                                  save_name=f'ceg_{name.lower()}.png')
                                  
        print(f"    Zone A: {z_pct['A']:.1f}%, Zone B: {z_pct['B']:.1f}%")
        print(f"    Zone C+D+E (Dangerous): {z_pct['C']+z_pct['D']+z_pct['E']:.2f}%")
