"""
step7_transformer.py — Transformer for Glucose Forecasting (Fixed)
==================================================================

Fixes from original:
1. Added Positional Encoding (crucial for time series in Transformers)
2. Stacked 3 encoder layers instead of 1
3. Added Cosine Annealing Learning Rate Scheduler
4. Removed the 50k sample cap (uses all data, ~200k+ windows)
"""

import numpy as np
import pandas as pd
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from step4_features import prepare_column

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension
    as the embeddings, so that the two can be summed.
    """
    def __init__(self, d_model, max_len=12):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class GlucoseTransformer(nn.Module):
    """
    Transformer encoder for glucose forecasting.

    Input:  (batch, seq_len, 1)  — 12 time-steps, 1 feature
    Output: (batch, 1)           — next glucose value
    """
    def __init__(self, seq_len=12, d_model=64, n_heads=4, num_layers=3, ff_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = self.input_proj(x)          # -> (batch, seq_len, d_model)
        x = self.pos_encoder(x)         # -> (batch, seq_len, d_model)
        x = self.transformer_encoder(x) # -> (batch, seq_len, d_model)
        
        # Use the representation of the LAST time step (or average pooling)
        # We'll use the last time step as it contains the most recent context
        x = x[:, -1, :]                 # -> (batch, d_model)
        
        return self.head(x).squeeze(-1) # -> (batch,)


def run_transformer(df, col_name, epochs=20, patience=3, batch_size=4096, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = prepare_column(df, col_name)

    # Scale to [0,1]
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    # Flatten X to scale, then reshape
    orig_shape_tr = X_train.shape
    orig_shape_te = X_test.shape
    
    X_train_s = sx.fit_transform(X_train.reshape(-1, 1)).reshape(orig_shape_tr)
    X_test_s  = sx.transform(X_test.reshape(-1, 1)).reshape(orig_shape_te)
    y_train_s = sy.fit_transform(y_train.reshape(-1, 1)).flatten()

    # PyTorch tensors — shape (N, 12, 1)
    X_tr = torch.tensor(X_train_s, dtype=torch.float32).unsqueeze(-1)
    X_te = torch.tensor(X_test_s,  dtype=torch.float32).unsqueeze(-1)
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

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = GlucoseTransformer().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val, best_weights, wait = float('inf'), None, 0

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    for epoch in range(epochs):
        # -- train --
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        
        scheduler.step()

        # -- validate --
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    model.eval()
    
    # Predict in batches to save memory
    test_loader = DataLoader(TensorDataset(X_te), batch_size=4096, shuffle=False)
    y_pred_s_list = []
    with torch.no_grad():
        for xb, in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
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

    print("Fixed Transformer Results (PyTorch):")
    print("-" * 45)

    results = {}
    for col in cols:
        print(f"  Training on: {col} ...", end=' ', flush=True)
        rmse, y_test, y_pred = run_transformer(df, col)
        results[col] = rmse
        print(f"RMSE: {rmse:.4f} mg/dL")

    best_col = min(results, key=results.get)
    _, y_test, y_pred = run_transformer(df, best_col)

    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:300], 'k-', label='True', linewidth=1.5)
    plt.plot(y_pred[:300], 'r--', label='Transformer', linewidth=1.5)
    plt.title(f'Fixed Transformer Best - {best_col} (RMSE: {results[best_col]:.2f})')
    plt.xlabel('Time Steps')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/transformer_fixed_best.png', dpi=150)
    plt.close()

    pd.DataFrame(list(results.items()), columns=['preprocessing', 'transformer_rmse']).to_csv('results/transformer_fixed_results.csv', index=False)
    print("\nSaved: results/transformer_fixed_results.csv")