import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves plots without opening a window
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────
# CONCEPT:
# We have a series with NaN gaps.
# Each method estimates what those missing values 
# should have been using different mathematical approaches.
# ─────────────────────────────────────────────────

def impute_hourly_mean(series, timestamps):
    """
    METHOD 1 — HOURLY MEAN
    
    Concept: Replace each missing value with the average 
    glucose for that hour of day across the entire dataset.
    
    Weakness: Ignores local context completely.
    If glucose was rising before the gap, this won't capture that.
    """
    result = series.copy()
    ts = pd.to_datetime(timestamps)
    hours = ts.hour
    
    # Compute mean per hour using non-missing values
    hourly_means = {}
    for h in range(24):
        mask = (hours == h) & (~series.isna())
        hourly_means[h] = (series[mask].mean() 
                           if mask.any() else series.mean())
    
    # Fill missing
    missing_idx = result[result.isna()].index
    for idx in missing_idx:
        h = hours[idx]
        result[idx] = hourly_means[h]
    
    return result


def impute_zoh(series):
    """
    METHOD 0 — ZERO ORDER HOLD (ZOH) / LOCF
    
    Concept: Last-Observation-Carried-Forward. Naively fill 
    the gap with the last known value.
    """
    return series.ffill().bfill()


def impute_linear(series):
    """
    METHOD 2 — LINEAR INTERPOLATION
    
    Concept: Draw a straight line between the last known 
    value before a gap and the first known value after it.
    Fill the gap with equally spaced points on that line.
    
    Weakness: Assumes constant rate of change.
    Real glucose curves are curved, not straight.
    """
    return series.interpolate(method='linear', 
                              limit_direction='both')


def impute_polynomial(series):
    """
    METHOD 3 — POLYNOMIAL (CUBIC) INTERPOLATION
    
    Concept: Fit ONE smooth polynomial curve through 
    all known surrounding data points, then use it 
    to estimate the missing values.
    
    Weakness: Runge's Phenomenon — a single high-degree
    polynomial oscillates wildly near the edges of the 
    interval. Can produce unrealistic glucose values.
    """
    values = series.values.astype(float)
    idx = np.arange(len(values))
    known = ~np.isnan(values)
    
    if known.sum() < 4:
        return impute_linear(series)
    
    # CubicSpline: fits one cubic polynomial globally
    cs = CubicSpline(idx[known], values[known], 
                     extrapolate=True)
    result = series.copy()
    result.iloc[~known] = cs(idx[~known])
    
    # Clip to physiological glucose range
    return result.clip(40, 400)


def impute_spline(series):
    """
    METHOD 4 — SPLINE INTERPOLATION  ← KEY METHOD
    
    Concept: Divide the interval into sub-intervals at 
    'knot' points. Fit a SEPARATE low-degree polynomial 
    (cubic = degree 3) on each sub-interval. Force all 
    pieces to connect smoothly at every knot:
      - Same value at the knot (continuity)
      - Same slope at the knot (first derivative match)
      - Same curvature at the knot (second derivative match)
    
    Why better than polynomial:
      - No Runge oscillation (each piece is only degree 3)
      - Globally smooth (smooth connections at knots)
      - Physiologically realistic (glucose is smooth)
    
    s=0 means INTERPOLATING spline:
    passes exactly through every known data point.
    """
    values = series.values.astype(float)
    idx = np.arange(len(values))
    known = ~np.isnan(values)
    
    if known.sum() < 5:
        return impute_linear(series)
    
    # UnivariateSpline with s=0: interpolating spline
    # k=3: cubic (degree 3) pieces
    # ext=3: extrapolate using boundary value
    spline = UnivariateSpline(
        idx[known], values[known], 
        k=3, s=0, ext=3
    )
    result = series.copy()
    result.iloc[~known] = spline(idx[~known])
    
    return result.clip(40, 400)


def apply_all_imputations(df):
    """Apply all 4 methods. Returns dict of series."""
    series = df['glucose_gaps'].copy()
    series.index = range(len(series))
    timestamps = df['timestamp'].values
    
    print("Applying imputation methods...")
    results = {
        'zoh':         impute_zoh(series),
        'hourly_mean': impute_hourly_mean(series, timestamps),
        'linear':      impute_linear(series),
        'polynomial':  impute_polynomial(series),
        'spline':      impute_spline(series),
        'true':        df['glucose'].values
    }
    print("Done.")
    return results


def plot_imputation(df, results, window=(200, 280)):
    """Plot all methods on a zoomed window around a gap."""
    start, end = window
    idx = range(start, end)
    ts = pd.to_datetime(df['timestamp'].values[start:end])
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts, results['true'][start:end], 
            'k-', linewidth=2.5, label='True Glucose', zorder=5)
    
    colors = {'zoh': 'purple', 'hourly_mean': 'orange', 'linear': 'blue',
              'polynomial': 'green', 'spline': 'red'}
    styles = {'zoh': '--', 'hourly_mean': '--', 'linear': '--',
              'polynomial': '--', 'spline': '--'}
    
    for method, color in colors.items():
        vals = results[method].values if hasattr(
            results[method], 'values') else results[method]
        ax.plot(ts, vals[start:end], 
                styles[method], color=color,
                linewidth=1.5, label=method.replace('_',' ').title(),
                alpha=0.85)
    
    # Shade missing regions
    missing = df['is_missing'].values[start:end]
    in_gap = False
    gap_start = None
    for i, m in enumerate(missing):
        if m and not in_gap:
            gap_start = ts[i]
            in_gap = True
        elif not m and in_gap:
            ax.axvspan(gap_start, ts[i], alpha=0.15, 
                      color='yellow', label='_gap')
            in_gap = False
    
    ax.set_title('Imputation Methods — Zoomed View Around Gaps')
    ax.set_xlabel('Time')
    ax.set_ylabel('Blood Glucose (mg/dL)')
    ax.legend(loc='upper right')
    ax.axhline(70, color='blue', alpha=0.3, linestyle=':')
    ax.axhline(180, color='red', alpha=0.3, linestyle=':')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/imputation_comparison.png', dpi=150)
    plt.close()
    print("Saved: results/imputation_comparison.png")


if __name__ == '__main__':
    import os
    df = pd.read_csv('data/hupa_combined.csv', 
                     parse_dates=['timestamp'])
    results = apply_all_imputations(df)
    plot_imputation(df, results)
    
    # Save imputed versions
    df['imputed_zoh']         = results['zoh'].values
    df['imputed_hourly_mean'] = results['hourly_mean'].values
    df['imputed_linear']      = results['linear'].values
    df['imputed_polynomial']  = results['polynomial'].values
    df['imputed_spline']      = results['spline'].values
    df.to_csv('data/hupa_imputed.csv', index=False)
    print("Saved: data/hupa_imputed.csv")