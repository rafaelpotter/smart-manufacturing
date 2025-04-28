# --- Imports ---
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
# We don't need 3D or scipy imports anymore for the requested plots

print(f"TensorFlow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Matplotlib Version: {plt.matplotlib.__version__}")


# --- Configuration ---
EXCEL_FILE_PATH = 'Surface roughness.xlsx' # Make sure this training file exists
TARGET_COLUMN = 'Ra value (µm)'      # <<< USE EXACT NAME FROM TRAINING FILE
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 8
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mean_squared_error'
METRICS = ['mae', 'mse']

# --- Prediction Configuration ---
PREDICTION_FILE_PATH = 'new_data_for_prediction.xlsx' # Make sure this prediction file exists or set to None
# PREDICTION_FILE_PATH = None

# --- Variations to Test ---
test_sizes_to_try = [0.2, 0.3]
activations_to_try = ['relu', 'tanh']

# --- Define Model Architectures ---
base_model_configs = [
    {'name': 'Simple_2L_64_32', 'layers': [64, 32]},
    {'name': 'Deeper_3L_128_64_32', 'layers': [128, 64, 32]},
    {'name': 'Wider_1L_128', 'layers': [128]},
    {'name': 'Smaller_2L_32_16', 'layers': [32, 16]},
    {'name': 'Medium_2L_64_64', 'layers': [64, 64]},
]

# --- 1. Load Training Data ---
print(f"\n--- Loading Training Data from {EXCEL_FILE_PATH} ---")
if not os.path.exists(EXCEL_FILE_PATH):
    print(f"Error: Training data file not found at {EXCEL_FILE_PATH}")
    exit()
try:
    df_full = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Training data loaded successfully. Shape: {df_full.shape}")
except Exception as e:
    print(f"Error reading training Excel file: {e}")
    exit()

# --- 2. Prepare Features (using the full original dataset) ---
print("\n--- Preparing Features from Full Dataset ---")
if TARGET_COLUMN not in df_full.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in training data.")
    exit()

X_full = df_full.drop(TARGET_COLUMN, axis=1)
y_full = df_full[TARGET_COLUMN]

# Handle non-numeric features (Drop)
numeric_cols = X_full.select_dtypes(include=np.number).columns
non_numeric_cols = X_full.select_dtypes(exclude=np.number).columns
if not non_numeric_cols.empty:
    print(f"Warning: Non-numeric features DROPPED from training data: {list(non_numeric_cols)}")
    X_full = X_full[numeric_cols].copy()
else:
    X_full = X_full[numeric_cols].copy()

if X_full.empty:
    print("Error: No numeric features remain in training data.")
    exit()
print(f"Using {X_full.shape[1]} numeric features for training: {list(X_full.columns)}")
training_feature_names = list(X_full.columns) # Store for prediction alignment


# --- Function to Build a Model ---
def build_model(input_shape, config):
    """Builds a Keras Sequential model based on the configuration."""
    model = tf.keras.models.Sequential(name=config['name_unique'])
    model.add(tf.keras.layers.Input(shape=(input_shape,), name='Input_Layer'))
    for i, num_neurons in enumerate(config['layers']):
        model.add(tf.keras.layers.Dense(num_neurons,
                                        activation=config['activation'],
                                        name=f'Hidden_{i+1}_{num_neurons}_{config["activation"]}'))
    model.add(tf.keras.layers.Dense(1, name='Output_Layer'))
    return model

# --- 3. Loop through Variations, Train and Evaluate ---
print("\n--- Starting Experiment Loop ---")
all_results = []
best_model_retrained = None
full_data_scaler = None

for test_size in test_sizes_to_try:
    print(f"\n===== Processing Test Size: {test_size:.1f} =====")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f"Train/Test Split: {X_train.shape[0]}/{X_test.shape[0]}")

    split_scaler = StandardScaler()
    X_train_scaled = split_scaler.fit_transform(X_train)
    X_test_scaled = split_scaler.transform(X_test)
    input_features = X_train_scaled.shape[1]

    for base_config in base_model_configs:
        for activation in activations_to_try:
            current_config = base_config.copy()
            current_config['activation'] = activation
            current_config['name_unique'] = f"{base_config['name']}_{activation}_ts{int(test_size*100)}"
            model_name = current_config['name_unique']
            print(f"\n--- Training Model: {model_name} ---")

            model = build_model(input_features, current_config)
            model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

            start_time = time.time()
            history = model.fit(X_train_scaled, y_train,
                                epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_split=0.1, verbose=0)
            training_time = time.time() - start_time
            print(f"Training finished in {training_time:.2f} seconds.")

            print(f"Evaluating model '{model_name}'...")
            loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=0)
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)
            print(f"  Test Loss (MSE): {loss:.4f}")
            print(f"  Test MAE: {mae:.4f}")
            print(f"  Test R-squared: {r2:.4f}")

            all_results.append({
                'model_name': base_config['name'], 'activation': activation,
                'test_size': test_size, 'full_name': model_name,
                'loss': loss, 'mae': mae, 'r2': r2, 'time': training_time,
                'layers': str(current_config['layers']), 'epochs_run': EPOCHS
            })
            tf.keras.backend.clear_session()
            del model, history

# --- 4. Consolidate and Display Results ---
print("\n\n--- Experiment Finished. Consolidating Results ---")
if not all_results:
    print("No models were trained. Exiting.")
    exit()
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by='r2', ascending=False)
print("\n--- Full Experiment Results (Sorted by Best R2 Overall) ---")
display_cols = ['r2', 'mae', 'loss', 'test_size', 'model_name', 'activation', 'layers', 'time', 'full_name']
print(results_df[display_cols].to_string())

# --- 5. Identify Best Model Config ---
print("\n--- Identifying Best Model Based on R2 Score ---")
best_config = None
if not results_df.empty:
    best_result = results_df.iloc[0]
    print(f"Best Model Found: {best_result['full_name']}")
    print(f"  R2 Score: {best_result['r2']:.4f}")
    best_config = {}
    for base_cfg in base_model_configs:
        if base_cfg['name'] == best_result['model_name']:
            best_config['name'] = base_cfg['name']
            break
    best_config['activation'] = best_result['activation']
    try:
        best_config['layers'] = ast.literal_eval(best_result['layers'])
        best_config['name_unique'] = f"BEST_MODEL_{best_config.get('name', 'Unk')}_{best_config.get('activation', 'unk')}_Retrained"
    except Exception as e:
        print(f"Error parsing layers string: {e}. Cannot proceed.")
        best_config = None
else:
    print("Results DataFrame is empty, cannot find best model.")

# --- 6. Retrain Best Model on Full Data ---
if best_config:
    print(f"\n--- Retraining Best Model ({best_config.get('name_unique', 'N/A')}) on FULL Dataset ---")
    print("Fitting Scaler on the entire dataset...")
    full_data_scaler = StandardScaler()
    X_full_scaled = full_data_scaler.fit_transform(X_full)
    print("Full dataset scaled.")

    best_model_retrained = build_model(X_full_scaled.shape[1], best_config)
    best_model_retrained.summary()
    best_model_retrained.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    print("Best model compiled. Training...")
    start_retrain_time = time.time()
    history_retrained = best_model_retrained.fit(X_full_scaled, y_full,
                                                 epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print(f"Retraining finished in {time.time() - start_retrain_time:.2f} seconds.")
else:
    print("\n--- Retraining Skipped: No valid best model configuration found. ---")

# --- 7. Prediction using the Retrained Best Model ---
if PREDICTION_FILE_PATH and best_model_retrained and full_data_scaler:
    print(f"\n--- Making Predictions using Retrained Best Model ---")
    if not os.path.exists(PREDICTION_FILE_PATH):
        print(f"Error: Prediction file not found at {PREDICTION_FILE_PATH}")
    else:
        try:
            print(f"Loading new data for prediction from: {PREDICTION_FILE_PATH}")
            df_predict = pd.read_excel(PREDICTION_FILE_PATH)
            print(f"Prediction data loaded. Shape: {df_predict.shape}")
            df_predict_original = df_predict.copy()

            print("Preprocessing prediction data...")
            missing_cols = [col for col in training_feature_names if col not in df_predict.columns]
            if missing_cols:
                print(f"Error: Prediction data is missing required columns: {missing_cols}")
            else:
                pred_numeric_cols = df_predict.select_dtypes(include=np.number).columns
                pred_non_numeric_cols = df_predict.select_dtypes(exclude=np.number).columns
                if not pred_non_numeric_cols.empty:
                    print(f"Warning: Non-numeric features DROPPED from prediction data: {list(pred_non_numeric_cols)}")
                    df_predict = df_predict[pred_numeric_cols]

                try:
                    df_predict = df_predict[training_feature_names] # Align columns
                    print("Prediction data columns aligned with training data.")
                    X_predict_scaled = full_data_scaler.transform(df_predict) # Scale
                    print("Prediction data scaled.")

                    print("Generating predictions...")
                    predictions = best_model_retrained.predict(X_predict_scaled, verbose=0).flatten()
                    print("Predictions generated.")
                    predicted_ra_col_name = f'Predicted_{TARGET_COLUMN}'
                    df_predict_original[predicted_ra_col_name] = predictions

                    # --- Display Prediction Table ---
                    print("\n--- Prediction Results ---")
                    print(df_predict_original.to_string())

                    # --- Generate ONLY the Requested Plot for Predicted Data ---
                    print("\n--- Generating Plot for Predicted Data ---")
                    if predictions.size > 0:

                        # --- Define Prediction Column Names based on User Input ---
                        # <<< ADJUST THESE NAMES TO MATCH YOUR PREDICTION FILE HEADERS EXACTLY >>>
                        cutting_speed_col = 'V (m/min)'
                        feed_rate_col = 'f (mm/tooth)'
                        # rake_angle_col = 'γ (°)' # Not needed for the current plot, but keep for reference
                        # <<< --- >>>

                        # --- Plot: Predicted Ra vs. Cutting Speed ---
                        if cutting_speed_col in df_predict_original.columns:
                            print(f"Generating Plot: Predicted Ra vs. {cutting_speed_col}")
                            plt.figure(figsize=(9, 6))
                            # Determine color feature and map (use Feed Rate if available)
                            color_feature = None
                            cmap_plot = None
                            label_plot = None
                            if feed_rate_col in df_predict_original.columns:
                                color_feature = df_predict_original[feed_rate_col]
                                cmap_plot = 'viridis'
                                label_plot = f'{feed_rate_col}'
                            else:
                                color_feature = 'blue' # Default color if feed rate column not found

                            # Create the scatter plot
                            scatter_plot = plt.scatter(
                                df_predict_original[cutting_speed_col],
                                df_predict_original[predicted_ra_col_name],
                                c=color_feature,
                                cmap=cmap_plot,
                                s=50, alpha=0.8, edgecolors='k'
                            )
                            # Add labels and title
                            plt.title(f'Predicted Ra vs. {cutting_speed_col}', fontsize=14)
                            plt.xlabel(f'{cutting_speed_col}', fontsize=12)
                            plt.ylabel(f'Predicted {TARGET_COLUMN}', fontsize=12)
                            # Add color bar if coloring by feed rate
                            if cmap_plot:
                                plt.colorbar(scatter_plot, label=label_plot)
                            # Add grid and show
                            plt.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout()
                            plt.show()
                        else:
                            print(f"Skipping Plot: Column '{cutting_speed_col}' not found in prediction data.")

                    else:
                         print("No predictions available, skipping prediction plots.")

                except KeyError as e: print(f"Error: Column mismatch during prediction alignment. Missing: {e}")
                except Exception as e: print(f"Error during prediction step: {e}")

        except Exception as e: print(f"Error reading prediction Excel file: {e}")
elif not PREDICTION_FILE_PATH: print("\n--- Prediction Skipped: PREDICTION_FILE_PATH not set. ---")
elif not best_model_retrained: print("\n--- Prediction Skipped: Best model not retrained. ---")
elif not full_data_scaler: print("\n--- Prediction Skipped: Scaler for full data not available. ---")


# --- 8. Plot Top Performing Models Overall ---
print("\n--- Generating Summary Plot: Top Performing Models ---")
NUM_TOP_MODELS_TO_PLOT = 10
if 'results_df' not in locals() or results_df.empty:
    print("No experiment results found, cannot create summary plot.")
else:
    results_df_sorted = results_df.sort_values(by='r2', ascending=False)
    num_to_plot = min(NUM_TOP_MODELS_TO_PLOT, len(results_df_sorted))
    if num_to_plot > 0:
        top_models_df = results_df_sorted.head(num_to_plot)
        df_subset_h = top_models_df.sort_values(by='r2', ascending=True) # Sort ascending for plot
        plot_labels_h = df_subset_h['full_name']
        r2_values_h = df_subset_h['r2']

        print(f"\n--- Generating Plot: Top {num_to_plot} Models by R-squared ---")
        plt.figure(figsize=(10, max(5, num_to_plot * 0.5)))
        bars_h = plt.barh(plot_labels_h, r2_values_h, color='darkcyan', height=0.7)
        plt.title(f'Top {num_to_plot} Models by R-squared Score (Across All Experiments)', fontsize=15)
        plt.xlabel('R-squared Score (Higher is Better)', fontsize=12); plt.ylabel('Model Configuration (Name_Activation_TestSize%)', fontsize=12); plt.yticks(fontsize=10)
        min_r2_plot = df_subset_h['r2'].min(); max_r2_plot = df_subset_h['r2'].max()
        r2_range = max_r2_plot - min_r2_plot; padding = r2_range * 0.05 if r2_range > 0 else 0.02
        lower_limit = max(0, min_r2_plot - padding) if min_r2_plot >= 0 else min_r2_plot - padding
        upper_limit = max(1.0, max_r2_plot + padding * 2)
        plt.xlim(left=lower_limit, right=upper_limit)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        if max_r2_plot <= 1.05 and lower_limit < 1.0:
             plt.axvline(1.0, color='firebrick', linewidth=1, linestyle=':'); plt.text(1.0, -0.05, 'R2=1.0', color='firebrick', ha='center', va='top', fontsize=9, transform=plt.gca().get_xaxis_transform())
        for bar in bars_h:
            xval = bar.get_width(); plt.text(xval + (upper_limit - lower_limit) * 0.01, bar.get_y() + bar.get_height()/2.0, f'{xval:.4f}', va='center', ha='left', fontsize=9, weight='semibold')
        plt.tight_layout(); plt.show()
    else: print("No models available in results to plot.")

print("\n--- Script Finished ---")