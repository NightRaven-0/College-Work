# hybrid_sensex_safe.py
"""
Patched hybrid stacking script:
- Adds MA5, MA10, daily return features
- Validates terminal input (rejects or clips OOD)
- Retrains RF (tuned) on new features
- If stacked hybrid < MLNN performance, fallback to MLNN alone
- Saves plots/predictions to hybrid_models/
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

# ---------- CONFIG ----------
CSV_PATH = r"R:\VS CODE\Dataset\Sensex-dataset.csv"
TARGET_COL = "Close"
OUT_DIR = "hybrid_models"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
HOLDOUT_FRAC = 0.2
RANDOM_STATE = 42
EPOCHS = 60
BATCH = 16
VALIDATION_SPLIT = 0.1
# ----------------------------

def rmse(a,b): return np.sqrt(mean_squared_error(a,b))

# indicators & helpers
def compute_macd(close, short=12, long=26, signal=9):
    ema_s = close.ewm(span=short, adjust=False).mean()
    ema_l = close.ewm(span=long, adjust=False).mean()
    macd = ema_s - ema_l
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_sig, macd - macd_sig

def compute_wr(high, low, close, period=14):
    hh = high.rolling(window=period, min_periods=1).max()
    ll = low.rolling(window=period, min_periods=1).min()
    return (hh - close) / (hh - ll) * -100

# load data
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(CSV_PATH)
df_raw = pd.read_csv(CSV_PATH).copy()
print("Loaded:", df_raw.shape)

# create indicators: MACD, WILLR, MA5, MA10, daily return
df = df_raw.copy()
df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df[TARGET_COL])
df['WILLR'] = compute_wr(df['High'], df['Low'], df[TARGET_COL])
df['MA5'] = df[TARGET_COL].rolling(window=5, min_periods=1).mean()
df['MA10'] = df[TARGET_COL].rolling(window=10, min_periods=1).mean()
df['Close_lag1'] = df[TARGET_COL].shift(1)
df['Close_lag2'] = df[TARGET_COL].shift(2)
df['ret1'] = (df[TARGET_COL] / df['Close_lag1'] - 1).replace([np.inf, -np.inf], np.nan)

# target = next-day Close
df['target_next'] = df[TARGET_COL].shift(-1)

# drop NaNs introduced
df = df.dropna().reset_index(drop=True)
print("After FE:", df.shape)

FEATURE_COLS = ['Open','High','Low','Close_lag1','Close_lag2','MA5','MA10','ret1','MACD','MACD_signal','MACD_hist','WILLR']

X = df[FEATURE_COLS].values
y = df['target_next'].values.reshape(-1,1)

# time-series splits
n = len(X)
n_test = int(n * TEST_SIZE)
split_idx = n - n_test
if n_test < 1:
    raise RuntimeError("TEST_SIZE too large")

X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]

# holdout for meta
hold_idx = int(len(X_train_full) * (1 - HOLDOUT_FRAC))
X_train_base, X_hold = X_train_full[:hold_idx], X_train_full[hold_idx:]
y_train_base, y_hold = y_train_full[:hold_idx], y_train_full[hold_idx:]

print("Sizes -> train_base:", X_train_base.shape[0], "holdout:", X_hold.shape[0], "test:", X_test.shape[0])

# scale X and y (y scaling stabilizes NN)
scaler_X = StandardScaler().fit(X_train_base)
X_train_base_s = scaler_X.transform(X_train_base)
X_hold_s = scaler_X.transform(X_hold)
X_test_s = scaler_X.transform(X_test)
joblib.dump(scaler_X, os.path.join(OUT_DIR,"scaler_X.pkl"))

scaler_y = StandardScaler().fit(y_train_base)
y_train_base_s = scaler_y.transform(y_train_base).ravel()
y_hold_s = scaler_y.transform(y_hold).ravel()
y_test_s = scaler_y.transform(y_test).ravel()
joblib.dump(scaler_y, os.path.join(OUT_DIR,"scaler_y.pkl"))

# small MLNN
def build_mlnn(input_dim):
    m = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

mlnn = build_mlnn(X_train_base_s.shape[1])
es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
print("\nTraining MLNN...")
mlnn.fit(X_train_base_s, y_train_base_s,
         validation_split=VALIDATION_SPLIT,
         epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=2)
mlnn.save(os.path.join(OUT_DIR,"mlnn_regressor.keras"))

# RandomForest with slightly stronger regularization + sample leaf
print("\nTraining RandomForest (tuned)...")
rf = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=5,
                           random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_base_s, y_train_base_s)
joblib.dump(rf, os.path.join(OUT_DIR,"rf_regressor.pkl"))

# base preds on hold/test
mlnn_hold = mlnn.predict(X_hold_s).ravel()
rf_hold = rf.predict(X_hold_s)
mlnn_test = mlnn.predict(X_test_s).ravel()
rf_test = rf.predict(X_test_s).ravel()

X_meta_train = np.vstack([mlnn_hold, rf_hold]).T
X_meta_test  = np.vstack([mlnn_test, rf_test]).T

# meta learner (Ridge)
meta = Ridge(alpha=1.0)
meta.fit(X_meta_train, y_hold_s)
print("Meta weights:", meta.coef_, "intercept:", meta.intercept_)

y_pred_hybrid_s = meta.predict(X_meta_test)

# invert scaling
mlnn_test_inv = scaler_y.inverse_transform(mlnn_test.reshape(-1,1)).ravel()
rf_test_inv = scaler_y.inverse_transform(rf_test.reshape(-1,1)).ravel()
hyb_test_inv = scaler_y.inverse_transform(y_pred_hybrid_s.reshape(-1,1)).ravel()
y_test_inv = scaler_y.inverse_transform(y_test_s.reshape(-1,1)).ravel()

# metrics
def report(name, yt, yp):
    print(f"{name}: RMSE={rmse(yt,yp):.2f}  MAE={mean_absolute_error(yt,yp):.2f}  R2={r2_score(yt,yp):.4f}")

report("MLNN", y_test_inv, mlnn_test_inv)
report("RandomForest", y_test_inv, rf_test_inv)
report("Stacked Hybrid", y_test_inv, hyb_test_inv)

# safety rule: if hybrid R2 < MLNN R2, fallback to MLNN predictions
mlnn_r2 = r2_score(y_test_inv, mlnn_test_inv)
hyb_r2 = r2_score(y_test_inv, hyb_test_inv)
if hyb_r2 < mlnn_r2:
    print("\nWARNING: Hybrid worse than MLNN on test set. Falling back to MLNN for final outputs.")
    final_preds = mlnn_test_inv
    final_name = "MLNN (fallback)"
else:
    final_preds = hyb_test_inv
    final_name = "Stacked Hybrid"

# save predictions
dates = df.loc[split_idx:, 'Date'].reset_index(drop=True)
out_df = pd.DataFrame({
    "Date_t": dates,
    "y_next_true": y_test_inv,
    "mlnn_pred": mlnn_test_inv,
    "rf_pred": rf_test_inv,
    "hybrid_pred": hyb_test_inv,
    "final_used": final_preds
})
out_df.to_csv(os.path.join(OUT_DIR,"hybrid_predictions_safe.csv"), index=False)
print("Saved predictions to:", os.path.abspath(OUT_DIR))

# plotting (true vs preds)
plt.figure(figsize=(12,6))
plt.plot(out_df['Date_t'], out_df['y_next_true'], label='True', color='black', linewidth=2)
plt.plot(out_df['Date_t'], out_df['mlnn_pred'], label='MLNN', alpha=0.7)
plt.plot(out_df['Date_t'], out_df['rf_pred'], label='RF', alpha=0.7)
plt.plot(out_df['Date_t'], out_df['final_used'], label=f'Final ({final_name})', color='red', linewidth=2)
plt.xticks(rotation=45); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"pred_vs_true_safe.png"), dpi=200)
plt.close()

# residual hist
resid = out_df['y_next_true'] - out_df['final_used']
plt.figure(figsize=(7,4)); plt.hist(resid, bins=30, alpha=0.7); plt.axvline(0, color='k', linestyle='--')
plt.title('Residuals (True - FinalPred)'); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"residuals_safe.png"), dpi=200); plt.close()

print("Saved plots to:", os.path.abspath(OUT_DIR))

# ----- terminal next-day prediction (only ask for yesterday_close) -----
# we WILL validate input against training target distribution
y_train_flat = y_train_base.ravel()
y_mean = y_train_flat.mean()
y_std = y_train_flat.std()

def predict_next_from_close(yesterday_close):
    # basic validation/clamping
    if yesterday_close <= 0:
        raise ValueError("Yesterday's close must be > 0.")
    # if OOD by > 3 std, warn & clip to nearest bound (prevents insane extrapolation)
    lower = y_mean - 3*y_std
    upper = y_mean + 3*y_std
    if (yesterday_close < lower) or (yesterday_close > upper):
        print(f"WARNING: entered value {yesterday_close} is outside (mean ± 3σ). Clipping to [{lower:.0f},{upper:.0f}].")
        yesterday_close = max(min(yesterday_close, upper), lower)

    # use last row of raw df as base, replace Close with yesterday_close and recompute indicators/feats
    hist = df_raw.copy().reset_index(drop=True)
    last = hist.iloc[-1].copy()
    new_row = last.copy()
    new_row['Close'] = float(yesterday_close)
    # append using concat
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    # recompute indicators & lags
    hist['MACD'], hist['MACD_signal'], hist['MACD_hist'] = compute_macd(hist[TARGET_COL])
    hist['WILLR'] = compute_wr(hist['High'], hist['Low'], hist[TARGET_COL])
    hist['Close_lag1'] = hist[TARGET_COL].shift(1)
    hist['Close_lag2'] = hist[TARGET_COL].shift(2)
    hist['MA5'] = hist[TARGET_COL].rolling(window=5, min_periods=1).mean()
    hist['MA10'] = hist[TARGET_COL].rolling(window=10, min_periods=1).mean()
    hist['ret1'] = (hist[TARGET_COL] / hist['Close_lag1'] - 1).replace([np.inf,-np.inf],np.nan)

    feat = hist.iloc[-1][FEATURE_COLS]
    # coerce numeric and check NaN
    feat = pd.to_numeric(feat, errors='coerce')
    if feat.isna().any():
        raise ValueError("Insufficient history for indicators after appending yesterday_close.")
    feat_row = feat.values.reshape(1,-1)
    feat_s = scaler_X.transform(feat_row)

    p_mlnn_s = mlnn.predict(feat_s).ravel()[0]
    p_rf_s = rf.predict(feat_s)[0]
    p_hyb_s = meta.predict([[p_mlnn_s, p_rf_s]])[0]

    p_mlnn = scaler_y.inverse_transform([[p_mlnn_s]])[0,0]
    p_rf = scaler_y.inverse_transform([[p_rf_s]])[0,0]
    p_hyb = scaler_y.inverse_transform([[p_hyb_s]])[0,0]

    # if hybrid is worse on test, return MLNN (fallback)
    if hyb_r2 < mlnn_r2:
        final = p_mlnn
        which = "MLNN (fallback)"
    else:
        final = p_hyb
        which = "Hybrid"
    return {"mlnn_next": float(p_mlnn), "rf_next": float(p_rf), "hybrid_next": float(p_hyb), "final_used": which}

# interactive run
if __name__ == "__main__":
    try:
        v = float(input("Enter yesterday's Close (single number): ").strip())
        preds = predict_next_from_close(v)
        print("\nPredictions (next day):", preds)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)