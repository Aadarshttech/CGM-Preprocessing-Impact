import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from step4_features import prepare_column

# ─────────────────────────────────────────────────
# CONCEPT — 1D-CNN:
# Applies convolutional filters along the TIME axis.
# A filter of size 3 looks at 3 consecutive glucose 
# readings at a time and detects local patterns:
# rising trend, plateau, sharp drop, etc.
# Stacking two conv layers detects progressively 
# more abstract patterns.
# Needs input shape: (samples, 1, timesteps)  ← PyTorch channel-first
#
# NOTE: Uses PyTorch instead of TensorFlow because
# TensorFlow has no wheel for Python 3.14.
# Architecture is identical.
# ─────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    1D-CNN for glucose forecasting.
    
    Input:  (batch, 1, 12)  — 12 time-steps, 1 channel
    Output: (batch, 1)      — next glucose value
    """
    def __init__(self, seq_len=12):
        super().__init__()
        self.net = nn.Sequential(
            # Conv layer 1: detect local glucose patterns
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),              # seq_len → 6

            # Conv layer 2: detect higher-level patterns
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),                 # 32 × 6 = 192

            nn.Linear(32 * (seq_len // 2), 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_cnn(df, col_name, epochs=30, patience=10, batch_size=128, max_samples=50_000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = prepare_column(df, col_name)

    # Subsample training data so CPU training completes in reasonable time.
    # 50 000 samples ≈ 69 days of 5-min CGM readings — more than enough signal.
    if len(X_train) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), max_samples, replace=False)
        idx.sort()                      # keep temporal order
        X_train, y_train = X_train[idx], y_train[idx]

    # Scale to [0,1] — neural nets train better on small values
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    X_train_s = sx.fit_transform(X_train)
    X_test_s  = sx.transform(X_test)
    y_train_s = sy.fit_transform(y_train.reshape(-1, 1)).flatten()

    # PyTorch tensors — shape (N, 1, 12) for Conv1d channel-first
    X_tr = torch.tensor(X_train_s, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test_s,  dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train_s, dtype=torch.float32)

    # Split off 10% validation
    n_val   = int(len(X_tr) * 0.1)
    n_train = len(X_tr) - n_val
    X_val, y_val = X_tr[n_train:], y_tr[n_train:]
    X_tr,  y_tr  = X_tr[:n_train],  y_tr[:n_train]

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CNN1D().to(device)
    opt    = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best_val, best_weights, wait = float('inf'), None, 0

    for epoch in range(epochs):
        # ── train ──
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break  # early stopping

    model.load_state_dict(best_weights)
    X_val, y_val = X_val.to(device), y_val.to(device)
    model.eval()
    
    # Predict in batches to save memory
    test_loader = DataLoader(TensorDataset(X_te), batch_size=4096, shuffle=False)
    y_pred_s_list = []
    with torch.no_grad():
        for xb, in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy().flatten()
            y_pred_s_list.append(preds)
            
    y_pred_s = np.concatenate(y_pred_s_list).flatten()

    y_pred = sy.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse, y_test, y_pred


if __name__ == '__main__':
    df = pd.read_csv('data/hupa_smoothed.csv')
    os.makedirs('results', exist_ok=True)

    cols = [c for c in df.columns
            if '_kalman' in c or '_smoothing_spline' in c or '_none' in c]

    print("1D-CNN Results (PyTorch):")
    print("-" * 45)

    results = {}
    for col in cols:
        print(f"  Training on: {col} ...", end=' ', flush=True)
        rmse, y_test, y_pred = run_cnn(df, col)
        results[col] = rmse
        print(f"RMSE: {rmse:.4f} mg/dL")

    best_col = min(results, key=results.get)
    _, y_test, y_pred = run_cnn(df, best_col)

    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:300], 'k-', label='True', linewidth=1.5)
    plt.plot(y_pred[:300], 'g--', label='1D-CNN', linewidth=1.5)
    plt.title(f'1D-CNN Best Result -- {best_col} '
              f'(RMSE: {results[best_col]:.2f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/cnn_best.png', dpi=150)
    plt.close()

    pd.DataFrame(list(results.items()),
                 columns=['preprocessing', 'cnn_rmse']
                 ).to_csv('results/cnn_results.csv', index=False)
    print("\nSaved: results/cnn_results.csv")