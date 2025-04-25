import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import time
import ast # Needed to parse layer string safely

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

# --- Prediction Configuration ---
# Set to None or a valid path to an Excel file with new data (features only)
PREDICTION_FILE_PATH = 'new_data_for_prediction.xlsx'
# PREDICTION_FILE_PATH = None # Uncomment this line if you don't want to predict

# --- Variations to Test ---
test_sizes_to_try = [0.2, 0.3] # Example: Test 20% and 30% test sets
# You can add more activation functions supported by Keras ('sigmoid', 'elu', etc.)
activations_to_try = ['relu', 'tanh']

# --- Define Model Architectures ---
base_model_configs = [
    {'name': 'Simple_2L_64_32', 'layers': [64, 32]},
    {'name': 'Deeper_3L_128_64_32', 'layers': [128, 64, 32]},
    {'name': 'Wider_1L_128', 'layers': [128]},
    {'name': 'Smaller_2L_32_16', 'layers': [32, 16]},
    {'name': 'Medium_2L_64_64', 'layers': [64, 64]},
]

# --- 1. Load Data ---
print(f"\n--- Loading Training Data from {EXCEL_FILE_PATH} ---")
if not os.path.exists(EXCEL_FILE_PATH):
    print(f"Error: Training data file not found at {EXCEL_FILE_PATH}")
    exit()
try:
    df_full = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Training data loaded successfully. Shape: {df_full.shape}")
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# --- 2. Prepare Features (using the full original dataset) ---
print("\n--- Preparing Features from Full Dataset ---")
if TARGET_COLUMN not in df_full.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found.")
    exit()

X_full = df_full.drop(TARGET_COLUMN, axis=1)
y_full = df_full[TARGET_COLUMN]

# Handle non-numeric features (Drop)
numeric_cols = X_full.select_dtypes(include=np.number).columns
non_numeric_cols = X_full.select_dtypes(exclude=np.number).columns
if not non_numeric_cols.empty:
    print(f"Warning: Non-numeric features DROPPED from training data: {list(non_numeric_cols)}")
    X_full = X_full[numeric_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
else:
    # Ensure X_full has only the numeric columns if all were numeric initially
    X_full = X_full[numeric_cols].copy()

if X_full.empty:
    print("Error: No numeric features remain in training data.")
    exit()
print(f"Using {X_full.shape[1]} numeric features for training: {list(X_full.columns)}")
# Store the feature names used for training (important for prediction preprocessing)
training_feature_names = list(X_full.columns)


# --- Function to Build a Model ---
def build_model(input_shape, config):
    """Builds a Keras Sequential model based on the configuration."""
    model = tf.keras.models.Sequential(name=config['name_unique']) # Use unique name
    model.add(tf.keras.layers.Input(shape=(input_shape,), name='Input_Layer'))
    for i, num_neurons in enumerate(config['layers']):
        model.add(tf.keras.layers.Dense(num_neurons,
                                        activation=config['activation'], # Use activation from config
                                        name=f'Hidden_{i+1}_{num_neurons}_{config["activation"]}'))
    model.add(tf.keras.layers.Dense(1, name='Output_Layer')) # Single output neuron for regression
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
    # IMPORTANT: This scaler is temporary for this specific train/test split evaluation.
    # A separate scaler will be fit on the *full* data later if prediction is needed.
    split_scaler = StandardScaler()
    X_train_scaled = split_scaler.fit_transform(X_train) # Fit ONLY on this training data
    X_test_scaled = split_scaler.transform(X_test)     # Transform test data using the SAME split scaler
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

            # 3e. Train Model
            start_time = time.time()
            history = model.fit(X_train_scaled, y_train,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_split=0.1, # Validation within the current train set
                                verbose=0) # Keep verbose=0 for loop clarity
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
                'layers': str(current_config['layers']), # Store layers as string
                'epochs_run': EPOCHS # Store epochs used
            })
            # Clear memory (optional, can help in very long runs)
            tf.keras.backend.clear_session()
            del model
            del history


# --- 4. Consolidate and Display Results ---
print("\n\n--- Experiment Finished. Consolidating Results ---")

if not all_results:
    print("No models were trained. Exiting.")
    exit()

results_df = pd.DataFrame(all_results)
# Sort by R2 score descending to easily find the best overall
results_df = results_df.sort_values(by='r2', ascending=False)

print("\n--- Full Experiment Results (Sorted by Best R2 Overall) ---")
# Select and order columns for better readability
display_cols = ['r2', 'mae', 'loss', 'test_size', 'model_name', 'activation', 'layers', 'time', 'full_name']
print(results_df[display_cols].to_string())

# --- 5. Identify and Prepare the Best Model for Potential Prediction ---
print("\n--- Identifying Best Model Based on R2 Score ---")
best_result = results_df.iloc[0] # Get the top row (highest R2)
print(f"Best Model Found: {best_result['full_name']}")
print(f"  R2 Score: {best_result['r2']:.4f}")
print(f"  Activation: {best_result['activation']}")
print(f"  Layers: {best_result['layers']}")
print(f"  Achieved with Test Size: {best_result['test_size']:.1f}")

# Prepare configuration for the best model
best_config = {}
# Find the original base config to get the name
for base_cfg in base_model_configs:
    if base_cfg['name'] == best_result['model_name']:
        best_config['name'] = base_cfg['name']
        break
best_config['activation'] = best_result['activation']
# Safely parse the layer string back into a list of integers
try:
    best_config['layers'] = ast.literal_eval(best_result['layers'])
except Exception as e:
    print(f"Error parsing layers string '{best_result['layers']}': {e}")
    print("Cannot proceed with retraining.")
    best_config = None # Mark as invalid

if best_config:
    best_config['name_unique'] = f"BEST_MODEL_{best_config['name']}_{best_config['activation']}_Retrained"

    # --- 6. Retrain Best Model on Full Data ---
    print(f"\n--- Retraining Best Model ({best_config['name_unique']}) on FULL Dataset ---")

    # 6a. Scale the FULL dataset
    print("Fitting Scaler on the entire dataset...")
    full_data_scaler = StandardScaler()
    # Use X_full which contains only the numeric features identified earlier
    X_full_scaled = full_data_scaler.fit_transform(X_full)
    print("Full dataset scaled.")

    # 6b. Build the best model architecture
    best_model_retrained = build_model(X_full_scaled.shape[1], best_config)
    best_model_retrained.summary() # Show summary of the final model

    # 6c. Compile the best model
    best_model_retrained.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    print("Best model compiled.")

    # 6d. Train the best model on the full scaled dataset
    print(f"Training for {EPOCHS} epochs...")
    start_retrain_time = time.time()
    # Use verbose=1 to see progress for this final training
    history_retrained = best_model_retrained.fit(X_full_scaled, y_full,
                                                 epochs=EPOCHS,
                                                 batch_size=BATCH_SIZE,
                                                 verbose=1)
    end_retrain_time = time.time()
    print(f"Retraining finished in {end_retrain_time - start_retrain_time:.2f} seconds.")

    # Optional: Evaluate on the full dataset itself (will likely show very good results)
    # loss_full, mae_full, mse_full = best_model_retrained.evaluate(X_full_scaled, y_full, verbose=0)
    # print(f"Evaluation on FULL training data: Loss={loss_full:.4f}, MAE={mae_full:.4f}")

    # --- 7. Prediction using the Retrained Best Model ---
    if PREDICTION_FILE_PATH and best_model_retrained:
        print(f"\n--- Making Predictions using Retrained Best Model ---")
        if not os.path.exists(PREDICTION_FILE_PATH):
            print(f"Error: Prediction file not found at {PREDICTION_FILE_PATH}")
        else:
            try:
                print(f"Loading new data for prediction from: {PREDICTION_FILE_PATH}")
                df_predict = pd.read_excel(PREDICTION_FILE_PATH)
                print(f"Prediction data loaded. Shape: {df_predict.shape}")

                # Store original prediction data for final output
                df_predict_original = df_predict.copy()

                # --- Preprocess Prediction Data ---
                print("Preprocessing prediction data...")

                # Check for required columns
                missing_cols = [col for col in training_feature_names if col not in df_predict.columns]
                if missing_cols:
                    print(f"Error: Prediction data is missing required columns: {missing_cols}")
                else:
                    # Drop any non-numeric columns that might be in the new data
                    pred_numeric_cols = df_predict.select_dtypes(include=np.number).columns
                    pred_non_numeric_cols = df_predict.select_dtypes(exclude=np.number).columns
                    if not pred_non_numeric_cols.empty:
                        print(f"Warning: Non-numeric features DROPPED from prediction data: {list(pred_non_numeric_cols)}")
                        df_predict = df_predict[pred_numeric_cols]

                    # Ensure columns are in the same order as training data and only include those used for training
                    try:
                        df_predict = df_predict[training_feature_names]
                        print("Prediction data columns aligned with training data.")

                        # Scale the prediction data using the scaler fitted on the FULL training data
                        X_predict_scaled = full_data_scaler.transform(df_predict)
                        print("Prediction data scaled.")

                        # --- Make Predictions ---
                        print("Generating predictions...")
                        predictions = best_model_retrained.predict(X_predict_scaled, verbose=0).flatten()
                        print("Predictions generated.")

                        # --- Display Predictions ---
                        # Add predictions to the original prediction dataframe for context
                        df_predict_original[f'Predicted_{TARGET_COLUMN}'] = predictions
                        print("\n--- Prediction Results ---")
                        print(df_predict_original.to_string())

                    except KeyError as e:
                        print(f"Error: Column mismatch during prediction data alignment. Missing column: {e}")
                    except Exception as e:
                        print(f"An error occurred during prediction preprocessing or prediction: {e}")

            except Exception as e:
                print(f"Error reading prediction Excel file: {e}")
    elif not PREDICTION_FILE_PATH:
        print("\n--- Prediction Skipped: PREDICTION_FILE_PATH not set. ---")
    elif not best_model_retrained:
         print("\n--- Prediction Skipped: Best model could not be retrained. ---")

# --- 8. Plot Results (from experiment loop, separate plots per Test Size) ---
# (Keep the plotting code as it was, it visualizes the experiment results)
print("\n--- Plotting Experiment Results ---")

# Need to re-sort by test_size for plotting convenience if it was sorted only by r2 before
results_df_plot = results_df.sort_values(by=['test_size', 'r2'], ascending=[True, False])

for test_size in results_df_plot['test_size'].unique():
    print(f"\n--- Generating Plots for Test Size: {test_size:.1f} ---")
    df_subset = results_df_plot[results_df_plot['test_size'] == test_size].copy()
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
    ax1.set_ylim(bottom=min(0, df_subset['r2'].min() - 0.1), top=max(1.05, df_subset['r2'].max() + 0.05)) # Ensure top limit is at least 1.05
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * np.sign(yval) if yval != 0 else 0.01, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center', fontsize=8)

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
    for i, model_name_label in enumerate(plot_labels): # Renamed variable to avoid conflict
        plt.text(time_values.iloc[i] * 1.01, r2_values.iloc[i], model_name_label, fontsize=9)

    plt.title(f'Performance vs Training Time (Test Size = {test_size:.1f})')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('R-squared Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.ylim(bottom=min(-0.05, df_subset['r2'].min() - 0.1), top=max(1.05, df_subset['r2'].max() + 0.05)) # Adjusted limits

    plt.tight_layout()
    plt.show()

print("\n--- Script Finished ---")