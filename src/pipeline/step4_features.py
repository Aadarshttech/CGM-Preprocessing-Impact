import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────
# CONCEPT — SLIDING WINDOW:
# 
# We convert the time series into supervised learning.
# 
# For each position i in the series:
#   INPUT (X):  glucose[i], glucose[i+1], ..., glucose[i+11]
#               = last 12 readings = last 60 minutes
#   
#   TARGET (y): glucose[i+12+5]
#               = glucose 6 steps after the window ends
#               = 30 minutes into the future
#
# Example:
#   Window: [120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 144, 146]
#   Target: 155  (what glucose will be 30 min later)
#
# We slide this window one step at a time across the 
# entire time series to create thousands of training examples.
# ─────────────────────────────────────────────────

def make_windows(glucose_array, window_size=12, horizon=6):
    """
    Create (X, y) pairs using sliding window.
    
    window_size = 12 steps = 60 minutes of input history
    horizon     = 6 steps  = 30 minutes ahead to predict
    """
    X, y = [], []
    n = len(glucose_array)
    
    for i in range(n - window_size - horizon):
        window = glucose_array[i : i + window_size]
        target = glucose_array[i + window_size + horizon - 1]
        
        # Skip if any NaN remains
        if np.isnan(window).any() or np.isnan(target):
            continue
        
        X.append(window)
        y.append(target)
    
    return np.array(X), np.array(y)


def time_series_split(X, y, test_ratio=0.2):
    """
    CONCEPT — WHY NO SHUFFLE:
    In time series, future data cannot influence past predictions.
    If we shuffle and split randomly, readings from tomorrow 
    end up in training and readings from yesterday in test.
    This causes 'data leakage' — unrealistically good results.
    
    Correct approach: first 80% = train, last 20% = test.
    This mirrors real deployment: train on past, test on future.
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


def prepare_column(df, col_name):
    """Full pipeline for one preprocessed column."""
    arr = df[col_name].values.astype(float)
    X, y = make_windows(arr)
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = pd.read_csv('data/hupa_smoothed.csv')
    
    # Test on one column
    col = 'spline_kalman'
    X_train, X_test, y_train, y_test = prepare_column(df, col)
    
    print(f"Column: {col}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Sample window: {X_train[0]}")
    print(f"Sample target: {y_train[0]:.2f} mg/dL")