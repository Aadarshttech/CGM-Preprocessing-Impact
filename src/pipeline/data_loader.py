import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────
# CONCEPT:
# Each patient has one CSV file with 8 columns.
# We only need: time + glucose.
# We combine all 25 patients into one long series.
# Separator is semicolon (;) not comma.
# ─────────────────────────────────────────────────

def load_one_patient(filepath):
    """Load one HUPA-UCM patient CSV. Returns DataFrame."""
    df = pd.read_csv(filepath, sep=';')
    
    # Keep only what we need
    df = df[['time', 'glucose']].copy()
    df.columns = ['timestamp', 'glucose']
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove any duplicates
    df = df.drop_duplicates(subset='timestamp')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def load_all_patients(folder_path):
    """Load all 25 patients. Returns one combined DataFrame."""
    files = sorted([f for f in os.listdir(folder_path) 
                    if f.endswith('.csv')])
    
    print(f"Found {len(files)} patient files\n")
    all_dfs = []
    
    for fname in files:
        path = os.path.join(folder_path, fname)
        df = load_one_patient(path)
        patient_id = fname.replace('.csv', '')
        df['patient_id'] = patient_id
        all_dfs.append(df)
        print(f"  {patient_id}: {len(df)} readings | "
              f"glucose range: {df['glucose'].min():.0f}–"
              f"{df['glucose'].max():.0f} mg/dL")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined)}")
    print(f"Glucose missing: {combined['glucose'].isna().sum()}")
    return combined


def inject_missing(df, missing_rate=0.10, gap_length=6, seed=42):
    """
    CONCEPT:
    The preprocessed data has NO missing values.
    We artificially remove 10% of readings in chunks of 6
    (= 30-minute gaps) to simulate sensor dropout.
    This lets us test whether our imputation methods 
    can recover the true values.
    
    missing_rate: fraction to remove (0.10 = 10%)
    gap_length:   consecutive readings to remove per gap
                  6 readings × 5 min = 30 minute gap
    """
    np.random.seed(seed)
    df = df.copy()
    n = len(df)
    n_gaps = int((n * missing_rate) / gap_length)
    
    removed = set()
    for _ in range(n_gaps):
        for _ in range(200):  # max attempts
            start = np.random.randint(gap_length, n - gap_length)
            chunk = set(range(start, start + gap_length))
            if not chunk & removed:
                removed.update(chunk)
                break
    
    df['glucose_gaps'] = df['glucose'].copy()
    df.loc[list(removed), 'glucose_gaps'] = np.nan
    df['is_missing'] = df.index.isin(removed)
    
    actual = df['is_missing'].sum()
    print(f"Injected {actual} missing values "
          f"({actual/n*100:.1f}%) across ~{n_gaps} gaps")
    return df


if __name__ == '__main__':
    # Use path relative to the script location (script is in results/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FOLDER = os.path.join(base_dir, 'data', 'HUPA-UCM', 'Preprocessed')
    df = load_all_patients(FOLDER)
    
    # Inject missing gaps
    df = inject_missing(df, missing_rate=0.10, gap_length=6)
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/hupa_combined.csv', index=False)
    print("\nSaved: data/hupa_combined.csv")
    print(df[['timestamp','patient_id','glucose',
              'glucose_gaps','is_missing']].head(20))