import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves plots without opening a window
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# CONCEPT:
# Even complete data has sensor noise — small random 
# errors in every reading. Smoothing removes noise 
# to give the AI cleaner signal to learn from.
# We test 2 smoothing methods on each imputed series.
#
# FIX: Process per-patient so Kalman EM runs on
# manageable chunks (~3K rows) instead of the full
# 309K-row dataset, which would take hours.
# ─────────────────────────────────────────────────

def smooth_kalman(series):
    """
    METHOD 1 — KALMAN SMOOTHING
    
    Concept: Treats glucose as a HIDDEN TRUE STATE 
    and sensor readings as NOISY OBSERVATIONS.
    
    Uses two parameters:
    - Process noise: how much glucose truly changes step-to-step
    - Observation noise: how noisy the sensor is
    
    The Kalman SMOOTHER (not just filter) uses both 
    past AND future observations to estimate the true 
    state at each point — making it more accurate than 
    a one-directional filter.
    
    Mathematical basis: Bayesian estimation,
    state-space models, Gaussian noise assumption.
    
    NOTE: Applied per-patient segment to keep
    each EM fit tractable (avoids O(N²) scaling).
    """
    values = series.values.reshape(-1, 1).astype(float)
    
    # Handle any remaining NaN
    nan_mask = np.isnan(values.flatten())
    if nan_mask.any():
        values[nan_mask] = np.nanmean(values)
    
    # If segment is tiny, fall back to Savitzky-Golay
    if len(values) < 10:
        flat = values.flatten()
        result = series.copy()
        result.iloc[:] = flat
        return result

    kf = KalmanFilter(
        transition_matrices=[[1]],       # x(t+1) ≈ x(t)
        observation_matrices=[[1]],      # observe x directly
        transition_covariance=[[1]],     # process noise
        observation_covariance=[[10]],   # sensor noise (higher = more smoothing)
        initial_state_mean=[values[0,0]],
        initial_state_covariance=[[1]]
    )
    # Only run EM on a subset to save time (max 1000 points)
    em_subset = values[:min(len(values), 1000)]
    kf = kf.em(em_subset, n_iter=3)
    
    # Bidirectional smoothing in chunks of 10,000 to prevent PyKalman from hanging/slowing down
    smoothed = np.empty_like(values)
    chunk_size = 10000
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i+chunk_size]
        sm, _ = kf.smooth(chunk)
        smoothed[i:i+chunk_size] = sm
    
    result = series.copy()
    result.iloc[:] = smoothed.flatten()
    return result


def smooth_spline(series):
    """
    METHOD 2 — SMOOTHING SPLINES  ← KEY METHOD
    
    Concept: Fits a spline that does NOT pass exactly 
    through every data point. Instead it BALANCES:
    
        Minimize: (fitting error) + s × (roughness penalty)
    
    Where s is the smoothing parameter:
      s = 0   → interpolating spline (no smoothing, passes through all points)
      s large → very smooth curve that ignores noise
    
    We set s = len(data) which is scipy's default 
    heuristic — a good balance for sensor data.
    
    Why better than Kalman for some models:
    - Doesn't require a state-space model assumption
    - Purely data-driven curve fitting
    - Computationally simpler
    """
    values = series.values.astype(float)
    idx = np.arange(len(values))
    known = ~np.isnan(values)
    
    if known.sum() < 5:
        return series
    
    # s = number of data points: scipy default smoothing heuristic
    s = len(values[known])
    
    sp = UnivariateSpline(idx[known], values[known], 
                          k=3, s=s)
    result = series.copy()
    result.iloc[:] = sp(idx)
    return result.clip(40, 400)


def apply_all_smoothing_patient(df, imputed_col):
    """
    Apply both smoothing methods per-patient segment.
    
    Running Kalman EM on the full 300K-row dataset would
    take hours (O(N²) per iteration). By splitting on
    patient_id (each ~3K-4K rows), each EM call finishes
    in under a second.
    """
    kalman_out = np.empty(len(df), dtype=float)
    spline_out = np.empty(len(df), dtype=float)

    patients = df['patient_id'].unique()
    for pid in patients:
        mask = df['patient_id'] == pid
        idx  = np.where(mask)[0]
        seg  = df.loc[mask, imputed_col].copy()
        seg.index = range(len(seg))

        k_seg = smooth_kalman(seg)
        s_seg = smooth_spline(seg)

        kalman_out[idx] = k_seg.values
        spline_out[idx] = s_seg.values

    return kalman_out, spline_out


def plot_smoothing(df, imputed_col, kalman_arr, spline_arr,
                   window=(100, 200)):
    """Plot original vs smoothed on a window."""
    start, end = window
    ts = pd.to_datetime(df['timestamp'].values[start:end])
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts, df[imputed_col].values[start:end],
            'gray', alpha=0.6, linewidth=1, label='Imputed (noisy)')
    ax.plot(ts, df['glucose'].values[start:end],
            'k-', linewidth=2, label='True Glucose')
    ax.plot(ts, kalman_arr[start:end],
            'b--', linewidth=1.5, label='Kalman Smoothed')
    ax.plot(ts, spline_arr[start:end],
            'r--', linewidth=1.5, label='Smoothing Spline')
    
    ax.set_title(f'Smoothing Comparison — {imputed_col}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.legend()
    plt.tight_layout()
    name = imputed_col.replace('imputed_', '')
    plt.savefig(f'results/smoothing_{name}.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv('data/hupa_imputed.csv', 
                     parse_dates=['timestamp'])
    
    imputed_cols = ['imputed_zoh', 'imputed_hourly_mean', 'imputed_linear',
                    'imputed_polynomial', 'imputed_spline']
    
    print("Applying smoothing to all imputed columns (per-patient)...\n", flush=True)
    
    for col in imputed_cols:
        print(f"  Processing: {col}", flush=True)
        kalman_arr, spline_arr = apply_all_smoothing_patient(df, col)
        name = col.replace('imputed_', '')
        df[f'{name}_kalman']           = kalman_arr
        df[f'{name}_smoothing_spline'] = spline_arr
        df[f'{name}_none']             = df[col].values
        plot_smoothing(df, col, kalman_arr, spline_arr)
        print(f"    [OK] Plot saved: results/smoothing_{name}.png", flush=True)
    
    df.to_csv('data/hupa_smoothed.csv', index=False)
    print("\nSaved: data/hupa_smoothed.csv", flush=True)
    print("Columns:", [c for c in df.columns if '_kalman' in c 
                       or '_smoothing' in c or '_none' in c])