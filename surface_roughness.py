import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import time

print(f"TensorFlow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")

# --- Configuration ---
EXCEL_FILE_PATH = 'Surface roughness.xlsx' # Make sure this file exists
TARGET_COLUMN = 'Ra value (µm)'      # Name of the column you want to predict
RANDOM_STATE = 42                     # For reproducible splits
EPOCHS = 50                           # Number of training cycles (can adjust globally)
BATCH_SIZE = 8                        # Number of samples per gradient update
OPTIMIZER = 'adam'                    # Common optimizer
LOSS_FUNCTION = 'mean_squared_error'  # Standard for regression
METRICS = ['mae', 'mse']              # Metrics to track during training

# --- Variations to Test ---
test_sizes_to_try = [0.2, 0.3] # Example: Test 20% and 30% test sets
# You can add more activation functions supported by Keras ('sigmoid', 'elu', etc.)
activations_to_try = ['relu', 'tanh']

# --- Define Model Architectures ---
# Each config now explicitly includes activation for clarity
# We will iterate through desired activations for *each* base architecture below
base_model_configs = [
    {'name': 'Simple_2L_64_32', 'layers': [64, 32]},
    {'name': 'Deeper_3L_128_64_32', 'layers': [128, 64, 32]},
    {'name': 'Wider_1L_128', 'layers': [128]},
    {'name': 'Smaller_2L_32_16', 'layers': [32, 16]},
    {'name': 'Medium_2L_64_64', 'layers': [64, 64]},
]

# --- 1. Load Data ---
print(f"\n--- Loading Data from {EXCEL_FILE_PATH} ---")
if not os.path.exists(EXCEL_FILE_PATH):
    print(f"Error: Excel file not found at {EXCEL_FILE_PATH}")
    exit()
try:
    df_full = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Data loaded successfully. Shape: {df_full.shape}")
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# --- 2. Prepare Features (outside the loop) ---
print("\n--- Preparing Features ---")
if TARGET_COLUMN not in df_full.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found.")
    exit()

X_full = df_full.drop(TARGET_COLUMN, axis=1)
y_full = df_full[TARGET_COLUMN]

# Handle non-numeric features (Drop)
numeric_cols = X_full.select_dtypes(include=np.number).columns
non_numeric_cols = X_full.select_dtypes(exclude=np.number).columns
if not non_numeric_cols.empty:
    print(f"Warning: Non-numeric features DROPPED: {list(non_numeric_cols)}")
    X_full = X_full[numeric_cols]

if X_full.empty:
    print("Error: No numeric features remain.")
    exit()
print(f"Using {X_full.shape[1]} numeric features: {list(X_full.columns)}")


# --- Function to Build a Model ---
def build_model(input_shape, config):
    """Builds a Keras Sequential model based on the configuration."""
    model = tf.keras.models.Sequential(name=config['name_unique']) # Use unique name
    model.add(tf.keras.layers.Input(shape=(input_shape,), name='Input_Layer'))
    for i, num_neurons in enumerate(config['layers']):
        model.add(tf.keras.layers.Dense(num_neurons,
                                        activation=config['activation'], # Use activation from config
                                        name=f'Hidden_{i+1}_{num_neurons}_{config["activation"]}'))
    model.add(tf.keras.layers.Dense(1, name='Output_Layer'))
    return model

# --- 3. Loop through Variations, Train and Evaluate ---
print("\n--- Starting Experiment Loop ---")
all_results = [] # Store results from all runs

# Loop over different test set sizes
for test_size in test_sizes_to_try:
    print(f"\n===== Processing Test Size: {test_size:.1f} =====")

    # 3a. Data Splitting for the current test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f"Train/Test Split: {X_train.shape[0]}/{X_test.shape[0]}")

    # 3b. Scaling based on the current training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Fit ONLY on this training data
    X_test_scaled = scaler.transform(X_test)     # Transform test data using the SAME scaler
    input_features = X_train_scaled.shape[1]

    # Loop over base model architectures
    for base_config in base_model_configs:
        # Loop over activation functions for the current architecture
        for activation in activations_to_try:
            # Create a unique configuration for this specific run
            current_config = base_config.copy() # Avoid modifying the original dict
            current_config['activation'] = activation
            # Create a more descriptive unique name
            current_config['name_unique'] = f"{base_config['name']}_{activation}_ts{int(test_size*100)}"

            model_name = current_config['name_unique']
            print(f"\n--- Training Model: {model_name} ---")

            # 3c. Build Model
            model = build_model(input_features, current_config)

            # 3d. Compile Model
            model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
            # print(f"Compiled model '{model_name}'") # Less verbose

            # 3e. Train Model
            start_time = time.time()
            # Note: histories are not stored globally anymore to save memory,
            # but you could store them if needed for detailed epoch plots per run
            history = model.fit(X_train_scaled, y_train,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_split=0.1, # Validation within the current train set
                                verbose=0) # Keep verbose=0
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training finished in {training_time:.2f} seconds.")

            # 3f. Evaluate Model
            print(f"Evaluating model '{model_name}'...")
            loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=0)
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)

            print(f"  Test Loss (MSE): {loss:.4f}")
            print(f"  Test MAE: {mae:.4f}")
            print(f"  Test R-squared: {r2:.4f}")

            # 3g. Store Results
            all_results.append({
                'model_name': base_config['name'], # Base architecture name
                'activation': activation,
                'test_size': test_size,
                'full_name': model_name, # Unique name for this run
                'loss': loss,
                'mae': mae,
                'r2': r2,
                'time': training_time,
                'layers': str(current_config['layers']) # Store layers as string for info
            })

# --- 4. Consolidate and Display Results ---
print("\n\n--- Experiment Finished. Consolidating Results ---")

if not all_results:
    print("No models were trained. Exiting.")
    exit()

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=['test_size', 'r2'], ascending=[True, False]) # Sort primarily by test_size, then R2

print("\n--- Full Experiment Results ---")
# Select and order columns for better readability
display_cols = ['test_size', 'model_name', 'activation', 'layers', 'r2', 'mae', 'loss', 'time', 'full_name']
print(results_df[display_cols].to_string()) # Use to_string to print more rows/cols if needed

# --- 5. Plot Results (Separate plots per Test Size) ---
print("\n--- Plotting Results ---")

for test_size in results_df['test_size'].unique():
    print(f"\n--- Generating Plots for Test Size: {test_size:.1f} ---")
    df_subset = results_df[results_df['test_size'] == test_size].copy()
    # Use the unique full name for plotting labels within this subset
    df_subset = df_subset.sort_values(by='r2', ascending=False)
    plot_labels = df_subset['full_name']
    r2_values = df_subset['r2']
    loss_values = df_subset['loss']
    time_values = df_subset['time']

    # --- Bar Plots for this Test Size ---
    fig_bars, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, len(plot_labels)*1.2), 7)) # Dynamic width
    fig_bars.suptitle(f'Model Comparison for Test Size = {test_size:.1f}', fontsize=16, y=1.02)

    # Plot R-squared
    bars1 = ax1.bar(plot_labels, r2_values, color='mediumseagreen')
    ax1.set_title('R-squared Score (Higher is Better)')
    ax1.set_ylabel('R-squared')
    ax1.tick_params(axis='x', rotation=75, labelsize=9) # Rotate more if names are long
    ax1.set_ylim(bottom=min(0, df_subset['r2'].min() - 0.1), top=max(1, df_subset['r2'].max() + 0.1))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * np.sign(yval), f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center', fontsize=8)

    # Plot Loss (MSE)
    bars2 = ax2.bar(plot_labels, loss_values, color='lightcoral')
    ax2.set_title('Mean Squared Error (Loss - Lower is Better)')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=75, labelsize=9) # Rotate more if names are long
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval*1.01, f'{yval:.2f}', va='bottom', ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout considering suptitle
    plt.show()

    # --- Scatter Plot for this Test Size ---
    plt.figure(figsize=(max(10, len(plot_labels)*0.5), 7)) # Dynamic width
    plt.scatter(time_values, r2_values, s=100, alpha=0.7, edgecolors='k', c=r2_values, cmap='viridis') # Color by R2 score
    plt.colorbar(label='R-squared Score')

    # Add labels to points
    for i, model_name in enumerate(plot_labels):
        plt.text(time_values.iloc[i] * 1.01, r2_values.iloc[i], model_name, fontsize=9)

    plt.title(f'Performance vs Training Time (Test Size = {test_size:.1f})')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('R-squared Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.ylim(bottom=min(0, df_subset['r2'].min() - 0.1), top=max(1, df_subset['r2'].max() + 0.1))

    plt.tight_layout()
    plt.show()

print("\n--- Script Finished ---")