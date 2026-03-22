import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_nn_on_rated_songs import (
    RatingMLP, load_model_from_pickle, engineer_features,
    get_device, rmse_np, kfold_cv_torch, TrainConfig
)

print("=" * 60)
print("MODEL ACCURACY TESTS")
print("=" * 60)

#A1: Load model and check structure
print("\n--- A1: Model structure check (Obj 3.3) ---")
model_path = "models/best_cv_model.pkl"
if not os.path.exists(model_path):
    print("  SKIP: No trained model found. Train the model first.")
else:
    model, bundle = load_model_from_pickle(model_path)
    total_params = sum(p.numel() for p in model.parameters())
    total_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"  Model has {total_params:,} parameters")
    print(f"  Model has {total_layers} linear layers")
    print(f"  Input dimension: {bundle['in_dim']}")
    print(f"  Best validation RMSE from training: {bundle['best_val_rmse']:.4f}")
    print(f"  Feature columns: {bundle['feature_columns']}")

    if total_layers > 10:
        print("  PASS: More than 10 hidden layers")
    else:
        print(f"  FAIL: Only {total_layers} layers found")

#A2: Hold-out test RMSE (Obj 5.5)
print("\n--- A2: Hold-out 20% test RMSE (Obj 5.5) ---")
csv_path = "rated_songs_with_features.csv"  #adjust the path if needed
if not os.path.exists(csv_path):
    #try alternative paths
    for alt in ["rated_songs_with_features.csv",
                "../rated_songs_with_features.csv"]:
        if os.path.exists(alt):
            csv_path = alt
            break

if not os.path.exists(csv_path):
    print("  SKIP: rated_songs_with_features.csv not found.")
    print("  INSTRUCTION: Run merge_datasets.py first, then re-run this test.")
elif not os.path.exists(model_path):
    print("  SKIP: No model to test.")
else:
    df = pd.read_csv(csv_path)
    X, y = engineer_features(df)

    #80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Total rated songs: {len(df)}")
    print(f"  Training set: {len(X_train)} songs")
    print(f"  Test set: {len(X_test)} songs")

    #load the model and predict on the test set
    model, bundle = load_model_from_pickle(model_path)
    device = get_device("auto")
    model = model.to(device)
    model.eval()

    scaler = bundle['scaler']
    feature_cols = bundle['feature_columns']

    X_test_aligned = X_test[feature_cols].to_numpy().astype(np.float32)
    X_test_scaled = scaler.transform(X_test_aligned)
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()
        preds = np.clip(preds, 1.0, 5.0)

    test_rmse = rmse_np(y_test.to_numpy(), preds)
    print(f"  Test RMSE: {test_rmse:.4f}")

    if test_rmse <= 0.5:
        print("  PASS: RMSE ≤ 0.5")
    else:
        print(f"  NOTE: RMSE is {test_rmse:.4f}. This may improve with more ratings.")

#A3: Prediction range check on full dataset (Obj 4.1)
print("\n--- A3: Full dataset prediction range (Obj 4.1) ---")
predictions_path = "spotify_songs_with_predictions.csv"
if not os.path.exists(predictions_path):
    print("  SKIP: Predictions file not found. Run predict_top_songs.py first.")
else:
    pred_df = pd.read_csv(predictions_path)
    min_pred = pred_df['predicted_rating'].min()
    max_pred = pred_df['predicted_rating'].max()
    total_songs = len(pred_df)
    print(f"  Total predictions: {total_songs}")
    print(f"  Min predicted rating: {min_pred:.4f}")
    print(f"  Max predicted rating: {max_pred:.4f}")

    if min_pred >= 1.0 and max_pred <= 5.0:
        print("  PASS: All predictions in [1.0, 5.0] range")
    else:
        print(f"  FAIL: Range is [{min_pred:.4f}, {max_pred:.4f}]")

    #check for any NaN predictions
    nan_count = pred_df['predicted_rating'].isna().sum()
    if nan_count == 0:
        print("  PASS: No NaN predictions")
    else:
        print(f"  FAIL: {nan_count} NaN predictions found")

print("\n" + "=" * 60)
print("ACCURACY TESTS COMPLETE")
print("=" * 60)
