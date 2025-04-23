import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score # Import R-squared metric
import matplotlib.pyplot as plt
import os
import time # To time training

print(f"TensorFlow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")

# --- Configuration ---
EXCEL_FILE_PATH = 'dataFromOktem.xlsx' # Make sure this file exists
TARGET_COLUMN = 'Ra values (lm)'      # Name of the column you want to predict
TEST_SIZE = 0.2                       # 20% of data for testing
RANDOM_STATE = 42                     # For reproducible splits
EPOCHS = 50                           # Number of training cycles
BATCH_SIZE = 8                        # Number of samples per gradient update
OPTIMIZER = 'adam'                    # Common optimizer
LOSS_FUNCTION = 'mean_squared_error'  # Standard for regression
METRICS = ['mae', 'mse']              # Metrics to track during training

# --- 1. Load Data ---
print(f"\n--- Loading Data from {EXCEL_FILE_PATH} ---")
if not os.path.exists(EXCEL_FILE_PATH):
    print(f"Error: Excel file not found at {EXCEL_FILE_PATH}")
    exit()

try:
    df = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# --- 2. Data Preprocessing ---
print("\n--- Preprocessing Data ---")

# Check if target column exists
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the Excel file.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

# Separate features (X) and target (y)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Handle potential non-numeric features (Basic: Drop them for simplicity)
numeric_cols = X.select_dtypes(include=np.number).columns
non_numeric_cols = X.select_dtypes(exclude=np.number).columns
if not non_numeric_cols.empty:
    print(f"Warning: Non-numeric feature columns found and will be DROPPED: {list(non_numeric_cols)}")
    X = X[numeric_cols] # Keep only numeric columns

if X.empty:
    print("Error: No numeric feature columns remaining after dropping non-numeric ones.")
    exit()

print(f"Using {X.shape[1]} numeric features: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit ONLY on training data
X_test_scaled = scaler.transform(X_test)     # Transform test data

# --- 3. Define Model Architectures ---
# List of dictionaries, each defining a model configuration
model_configs = [
    {
        'name': 'Simple_2L_64_32', # Shortened names for plot labels
        'layers': [64, 32],
        'activation': 'relu'
    },
    {
        'name': 'Deeper_3L_128_64_32',
        'layers': [128, 64, 32],
        'activation': 'relu'
    },
    {
        'name': 'Wider_1L_128',
        'layers': [128],
        'activation': 'relu'
    },
    {
        'name': 'Smaller_2L_32_16',
        'layers': [32, 16],
        'activation': 'relu'
    },
    {
        'name': 'Medium_2L_64_64',
        'layers': [64, 64],
        'activation': 'relu'
    },
    # Add more configurations if needed
]

# --- Function to Build a Model ---
def build_model(input_shape, config):
    """Builds a Keras Sequential model based on the configuration."""
    model = tf.keras.models.Sequential(name=config['name'])
    model.add(tf.keras.layers.Input(shape=(input_shape,), name='Input_Layer'))
    for i, num_neurons in enumerate(config['layers']):
        model.add(tf.keras.layers.Dense(num_neurons,
                                        activation=config['activation'],
                                        name=f'Hidden_Layer_{i+1}_{num_neurons}'))
    model.add(tf.keras.layers.Dense(1, name='Output_Layer'))
    return model

# --- 4. Train and Evaluate Models ---
print("\n--- Training and Evaluating Models ---")
results = {} # To store evaluation results {model_name: {'loss': float, 'mae': float, 'r2': float, 'time': float}}
histories = {} # To store training history {model_name: history_object}

input_features = X_train_scaled.shape[1]

for config in model_configs:
    model_name = config['name']
    print(f"\n--- Training Model: {model_name} ---")

    # Build
    model = build_model(input_features, config)

    # Compile
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS_FUNCTION,
                  metrics=METRICS)
    print(f"Compiled model '{model_name}'")

    # Train
    start_time = time.time()
    history = model.fit(X_train_scaled, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        verbose=0) # Keep verbose=0 for cleaner loop output
    end_time = time.time()
    training_time = end_time - start_time
    histories[model_name] = history
    print(f"Training finished in {training_time:.2f} seconds.")

    # Evaluate
    print(f"Evaluating model '{model_name}' on test data...")
    loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Calculate R-squared score
    y_pred = model.predict(X_test_scaled, verbose=0).flatten() # Get predictions for R2 calc
    r2 = r2_score(y_test, y_pred)

    print(f"  Test Loss (MSE): {loss:.4f}")
    print(f"  Test MAE: {mae:.4f}")
    print(f"  Test R-squared: {r2:.4f}") # Print R-squared score

    # Store results
    results[model_name] = {
        'loss': loss,
        'mae': mae,
        'r2': r2, # Store R-squared
        'time': training_time
    }

# --- 5. Compare Results ---
print("\n--- Comparison of Model Results (Sorted by R-squared) ---")

# Create a DataFrame for easier viewing
results_df = pd.DataFrame(results).T # Transpose to have models as rows
results_df = results_df.sort_values(by='r2', ascending=False) # Sort by R-squared (higher is better)

print(results_df[['r2', 'mae', 'loss', 'time']]) # Display R2 first

# --- 6. Plot Comparison: R-squared and Loss ---
print("\n--- Plotting Comparison: R-squared & Loss ---")

model_names = results_df.index
r2_values = results_df['r2']
loss_values = results_df['loss'] # This is MSE

fig_bars, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Increased height slightly for rotated labels

# Plot R-squared (Higher is better)
bars1 = ax1.bar(model_names, r2_values, color='mediumseagreen')
ax1.set_title('Model Comparison: R-squared Score (Test Set)')
ax1.set_ylabel('R-squared (Higher is Better)')
ax1.set_xlabel('Model Architecture')
# Corrected tick_params: removed ha='right'
ax1.tick_params(axis='x', rotation=45, labelsize=9) # Rotate labels, slightly smaller font if needed
ax1.set_ylim(bottom=min(0, results_df['r2'].min() - 0.1), top=max(1, results_df['r2'].max() + 0.1))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
# Add text labels for R-squared values
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * np.sign(yval), f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center', fontsize=9)


# Plot Loss (MSE) (Lower is better)
bars2 = ax2.bar(model_names, loss_values, color='lightcoral')
ax2.set_title('Model Comparison: Mean Squared Error (Loss - Test Set)')
ax2.set_ylabel('MSE (Lower is Better)')
ax2.set_xlabel('Model Architecture')
# Corrected tick_params: removed ha='right'
ax2.tick_params(axis='x', rotation=45, labelsize=9) # Rotate labels, slightly smaller font if needed
ax2.grid(axis='y', linestyle='--', alpha=0.7)
# Add text labels for Loss values
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval*1.01, f'{yval:.2f}', va='bottom', ha='center', fontsize=9) # Adjust position slightly


plt.tight_layout(pad=1.5) # Adjust layout with some padding
plt.show()

# --- 7. Scatter Plot: R-squared vs. Training Time ---
# (This section should be okay, no changes needed here unless you want adjustments)
print("\n--- Plotting Scatter Plot: R-squared vs Training Time ---")

time_values = results_df['time']

plt.figure(figsize=(10, 7))
plt.scatter(time_values, r2_values, s=100, alpha=0.7, edgecolors='k') # s is marker size

# Add labels to points
for i, model_name in enumerate(model_names):
    plt.text(time_values[i] * 1.01, r2_values[i], model_name, fontsize=9)

plt.title('Model Performance: R-squared vs. Training Time')
plt.xlabel('Training Time (seconds)')
plt.ylabel('R-squared Score (Test Set)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='grey', linewidth=0.8, linestyle='--') # Add line at R2=0
plt.ylim(bottom=min(0, results_df['r2'].min() - 0.1), top=max(1, results_df['r2'].max() + 0.1)) # Consistent Y axis with bar chart

plt.tight_layout()
plt.show()

# --- 8. Plot Training History (Optional) ---
# This section remains the same as before, plotting loss and MAE curves
# Useful for diagnosing overfitting/underfitting for each model individually

# num_models = len(histories)
# fig_hist, axes_hist = plt.subplots(num_models, 2, figsize=(12, 5 * num_models), squeeze=False)
# print("\n--- Plotting Training Histories ---")

# model_idx = 0
# for model_name, history in histories.items():
#     # Plot Loss
#     ax_loss = axes_hist[model_idx, 0]
#     ax_loss.plot(history.history['loss'], label='Training Loss (MSE)')
#     ax_loss.plot(history.history['val_loss'], label='Validation Loss (MSE)')
#     ax_loss.set_title(f'{model_name} - Loss')
#     ax_loss.set_ylabel('Loss (MSE)')
#     ax_loss.set_xlabel('Epoch')
#     ax_loss.legend()
#     ax_loss.grid(True)

#     # Plot MAE
#     ax_mae = axes_hist[model_idx, 1]
#     ax_mae.plot(history.history['mae'], label='Training MAE')
#     ax_mae.plot(history.history['val_mae'], label='Validation MAE')
#     ax_mae.set_title(f'{model_name} - MAE')
#     ax_mae.set_ylabel('Mean Absolute Error')
#     ax_mae.set_xlabel('Epoch')
#     ax_mae.legend()
#     ax_mae.grid(True)

#     model_idx += 1

# plt.tight_layout()
# plt.show()


print("\n--- Script Finished ---")