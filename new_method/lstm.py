"""
LSTM Deep Learning for RUL Prediction
Captures temporal dependencies in sequential acoustic emission data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow/Keras is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠ TensorFlow not installed. Install with: pip install tensorflow")

plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
FILE_PATH = r'F:\concrete data\test 3\per_file_features_800.csv'
OUTPUT_DIR = r'F:\concrete data\test 3\lstm_results'
DPI = 300

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not KERAS_AVAILABLE:
    print("\nPlease install TensorFlow to use LSTM models:")
    print("  pip install tensorflow")
    exit()

print("="*80)
print("LSTM DEEP LEARNING FOR RUL PREDICTION")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv(FILE_PATH)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# Create RUL target
df['RUL_percentage'] = 100 * (1 - df.index / (len(df) - 1))
TARGET = 'RUL_percentage'

# Select trending features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if 'RUL' not in col]

from scipy.stats import spearmanr
trend_features = []
for col in feature_cols:
    corr, p_val = spearmanr(df.index, df[col])
    if abs(corr) > 0.5 and p_val < 0.01:
        trend_features.append(col)

print(f"\nSelected {len(trend_features)} trending features")

# Prepare data
X = df[trend_features].fillna(df[trend_features].median())
y = df[TARGET]

# ============================================================================
# 2. CREATE SLIDING WINDOW SEQUENCES
# ============================================================================
print("\n[2] CREATING TIME-SERIES SEQUENCES...")

def create_sequences(X, y, sequence_length):
    """Create sliding window sequences for LSTM"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences (use last N points to predict next RUL)
SEQUENCE_LENGTH = 10  # Use last 10 segments to predict RUL
print(f"Sequence length: {SEQUENCE_LENGTH} segments")

X_sequences, y_sequences = create_sequences(X_scaled, y.values, SEQUENCE_LENGTH)

print(f"\nSequence shape: {X_sequences.shape}")
print(f"  - Total sequences: {X_sequences.shape[0]}")
print(f"  - Sequence length: {X_sequences.shape[1]}")
print(f"  - Features per step: {X_sequences.shape[2]}")

# Train/test split (80/20, respecting temporal order)
split_idx = int(0.8 * len(X_sequences))

X_train = X_sequences[:split_idx]
X_test = X_sequences[split_idx:]
y_train = y_sequences[:split_idx]
y_test = y_sequences[split_idx:]

print(f"\nTrain sequences: {len(X_train)}")
print(f"Test sequences: {len(X_test)}")

# ============================================================================
# 3. BUILD LSTM MODELS
# ============================================================================
print("\n" + "="*80)
print("BUILDING LSTM MODELS")
print("="*80)

def build_simple_lstm(input_shape):
    """Simple LSTM model"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bidirectional_lstm(input_shape):
    """Bidirectional LSTM for better context"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_deep_lstm(input_shape):
    """Deep LSTM with multiple layers"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build models
input_shape = (X_train.shape[1], X_train.shape[2])

models = {
    'Simple LSTM': build_simple_lstm(input_shape),
    'Bidirectional LSTM': build_bidirectional_lstm(input_shape),
    'Deep LSTM': build_deep_lstm(input_shape)
}

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=0
)

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n[3] TRAINING LSTM MODELS...")

results = {}
histories = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print('='*60)
    print(f"Parameters: {model.count_params():,}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    histories[name] = history
    
    # Predictions
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'history': history,
        'y_pred_test': y_pred_test,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Epochs': len(history.history['loss'])
    }
    
    print(f"\nResults:")
    print(f"  Epochs trained: {len(history.history['loss'])}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.2f}%")
    print(f"  Test MAE:  {test_mae:.2f}%")

# Save results
results_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k not in ['model', 'history', 'y_pred_test']}
    for name, vals in results.items()
}).T
results_df.to_csv(os.path.join(OUTPUT_DIR, 'lstm_comparison.csv'))
print("\n✓ Saved: lstm_comparison.csv")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Training History
fig, axes = plt.subplots(len(models), 2, figsize=(14, 4*len(models)))
if len(models) == 1:
    axes = axes.reshape(1, -1)

for idx, (name, result) in enumerate(results.items()):
    history = result['history']
    
    # Loss
    ax = axes[idx, 0]
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontweight='bold')
    ax.set_title(f'{name} - Training History', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE
    ax = axes[idx, 1]
    ax.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title(f'{name} - MAE History', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_training_history.png'), dpi=DPI, bbox_inches='tight')
print("✓ Saved: 1_training_history.png")
plt.close()

# 2. Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

model_names = list(results.keys())

# R² comparison
ax = axes[0, 0]
x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, [results[m]['Train_R2'] for m in model_names], 
       width, label='Train', alpha=0.8)
ax.bar(x + width/2, [results[m]['Test_R2'] for m in model_names], 
       width, label='Test', alpha=0.8)
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('Model Performance (R²)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Error metrics
ax = axes[0, 1]
errors = [[results[m]['Test_RMSE'], results[m]['Test_MAE']] for m in model_names]
x = np.arange(2)
width = 0.25
for i, name in enumerate(model_names):
    ax.bar(x + i*width, errors[i], width, label=name, alpha=0.8)
ax.set_ylabel('Error (%)', fontweight='bold')
ax.set_title('Error Metrics', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['RMSE', 'MAE'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Best model predictions
best_model_name = max(results, key=lambda x: results[x]['Test_R2'])
best_pred = results[best_model_name]['y_pred_test']

ax = axes[1, 0]
ax.scatter(y_test, best_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual RUL (%)', fontweight='bold')
ax.set_ylabel('Predicted RUL (%)', fontweight='bold')
ax.set_title(f'{best_model_name} - Predictions\nR² = {results[best_model_name]["Test_R2"]:.4f}',
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Timeline
ax = axes[1, 1]
test_indices = range(split_idx + SEQUENCE_LENGTH, len(df))
ax.plot(test_indices, y_test, 'o-', label='Actual RUL', linewidth=2, markersize=3)
ax.plot(test_indices, best_pred, 's-', label='Predicted RUL', 
        linewidth=2, markersize=3, alpha=0.7)
ax.fill_between(test_indices, y_test, best_pred, alpha=0.2)
ax.set_xlabel('Segment Number', fontweight='bold')
ax.set_ylabel('RUL (%)', fontweight='bold')
ax.set_title('RUL Prediction Timeline', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_lstm_comparison.png'), dpi=DPI, bbox_inches='tight')
print("✓ Saved: 2_lstm_comparison.png")
plt.close()

# 3. Residual Analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    residuals = y_test - result['y_pred_test']
    
    ax.scatter(result['y_pred_test'], residuals, alpha=0.5, s=30, 
               edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted RUL (%)', fontweight='bold')
    ax.set_ylabel('Residual (%)', fontweight='bold')
    ax.set_title(f'{name}\nMAE = {result["Test_MAE"]:.2f}%', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_residuals.png'), dpi=DPI, bbox_inches='tight')
print("✓ Saved: 3_residuals.png")
plt.close()

# ============================================================================
# 6. FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("GENERATING REPORT")
print("="*80)

report = f"""
LSTM DEEP LEARNING FOR RUL PREDICTION - REPORT
{'='*80}

DATASET:
  Total segments: {len(df)}
  Sequence length: {SEQUENCE_LENGTH} segments
  Features: {len(trend_features)}
  
LSTM CONFIGURATION:
  Train sequences: {len(X_train)}
  Test sequences: {len(X_test)}
  Input shape: {input_shape}

MODEL PERFORMANCE:
"""

for name, res in results.items():
    report += f"""
{name}:
  Parameters: {res['model'].count_params():,}
  Epochs trained: {res['Epochs']}
  Test R²: {res['Test_R2']:.4f}
  Test RMSE: {res['Test_RMSE']:.2f}%
  Test MAE: {res['Test_MAE']:.2f}%
"""

report += f"""
BEST MODEL: {best_model_name}
  ✓ Test R²: {results[best_model_name]['Test_R2']:.4f}
  ✓ Test MAE: {results[best_model_name]['Test_MAE']:.2f}%

ADVANTAGES OF LSTM:
  - Captures temporal dependencies in sequential data
  - Learns patterns across time steps
  - Handles variable-length sequences
  - Better for long-term predictions

COMPARISON TO TRADITIONAL ML:
  Traditional models predict from single time point
  LSTM uses sequence of {SEQUENCE_LENGTH} previous points
  More context → potentially better predictions

GENERATED FILES:
  1. lstm_comparison.csv: Model performance metrics
  2. 1_training_history.png: Training curves
  3. 2_lstm_comparison.png: Model comparison and predictions
  4. 3_residuals.png: Residual analysis
  5. lstm_rul_report.txt: This report

{'='*80}
LSTM analysis complete!
All results saved to: {OUTPUT_DIR}
{'='*80}
"""

print(report)

with open(os.path.join(OUTPUT_DIR, 'lstm_rul_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✓ Report saved to: {OUTPUT_DIR}/lstm_rul_report.txt")
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\n" + "="*80)
print("LSTM RUL PREDICTION COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"Test R²: {results[best_model_name]['Test_R2']:.4f}")
print(f"Test MAE: {results[best_model_name]['Test_MAE']:.2f}%")
print("="*80)