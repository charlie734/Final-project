import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_nn_on_rated_songs import engineer_features, RatingMLP, TrainConfig, fit_final_ensemble
from predict_top_songs import remove_duplicates
from merge_datasets import merge_datasets
from song_predictions_viewer import load_predictions

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} - {detail}")

print("=" * 60)
print("INTEGRATION TESTS")
print("=" * 60)

#I1: Merge -> Training pipeline (Obj 2.2, 3.1)
print("\n--- I1: Merge output feeds into engineer_features (Obj 2.2, 3.1) ---")
#create fake ratings and songs CSVs in a temp directory
tmp_dir = tempfile.mkdtemp()
original_dir = os.getcwd()
try:
    ratings_path = os.path.join(tmp_dir, "ratings.csv")
    songs_path = os.path.join(tmp_dir, "songs.csv")
    output_path = os.path.join(tmp_dir, "merged.csv")

    #create a small ratings file
    pd.DataFrame({
        'track_id': ['t1', 't2', 't3', 't4', 't5'],
        'rating': [5, 4, 3, 2, 1]
    }).to_csv(ratings_path, index=False)

    #create a small songs file with the required audio features
    pd.DataFrame({
        'track_id': ['t1', 't2', 't3', 't4', 't5', 't6'],
        'track_name': ['Song1', 'Song2', 'Song3', 'Song4', 'Song5', 'Song6'],
        'track_artist': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        'danceability': [0.5, 0.7, 0.3, 0.8, 0.4, 0.6],
        'energy': [0.8, 0.6, 0.4, 0.9, 0.3, 0.7],
        'speechiness': [0.05, 0.1, 0.03, 0.04, 0.08, 0.06],
        'acousticness': [0.2, 0.8, 0.5, 0.1, 0.9, 0.4],
        'instrumentalness': [0.0, 0.01, 0.5, 0.0, 0.7, 0.1],
        'liveness': [0.1, 0.3, 0.6, 0.2, 0.4, 0.5],
        'valence': [0.9, 0.4, 0.2, 0.7, 0.3, 0.6],
        'tempo': [120, 90, 150, 130, 100, 110],
        'duration_ms': [200000, 180000, 240000, 210000, 190000, 220000],
        'key': [5, 0, 11, 3, 7, 2],
        'mode': [1, 0, 1, 1, 0, 0],
        'loudness': [-5, -10, -3, -7, -12, -6],
        'time_signature': [4, 4, 3, 4, 4, 3],
        'track_popularity': [70, 30, 50, 80, 20, 60],
        'track_album_release_date': ['2020-01-01']*6
    }).to_csv(songs_path, index=False)

    #run merge
    merged = merge_datasets(ratings_path, songs_path, output_path)
    test("Merge produces 5 rows (inner join)", len(merged) == 5)
    test("Merged file contains 'rating' column", 'rating' in merged.columns)
    test("Merged file contains 'danceability'", 'danceability' in merged.columns)

    #feed into engineer_features
    X, y = engineer_features(merged)
    test("engineer_features accepts merge output", X.shape[0] == 5)
    test("Features have >7 columns", X.shape[1] > 7, f"got {X.shape[1]}")
    test("Target y has 5 values", len(y) == 5)

finally:
    shutil.rmtree(tmp_dir)

#I2: Training -> Prediction pipeline (Obj 3.7, 4.1)
print("\n--- I2: Ensemble predict accepts engineer_features output (Obj 3.7, 4.1) ---")
#create a small dataset and train a tiny ensemble
test_df = pd.DataFrame({
    'danceability': np.random.rand(20),
    'energy': np.random.rand(20),
    'speechiness': np.random.rand(20),
    'acousticness': np.random.rand(20),
    'instrumentalness': np.random.rand(20),
    'liveness': np.random.rand(20),
    'valence': np.random.rand(20),
    'tempo': np.random.rand(20) * 200,
    'duration_ms': np.random.randint(100000, 300000, 20),
    'key': np.random.randint(0, 12, 20),
    'mode': np.random.randint(0, 2, 20),
    'loudness': np.random.rand(20) * -20,
    'time_signature': [4]*20,
    'track_popularity': np.random.randint(0, 100, 20),
    'track_album_release_date': ['2020-01-01']*20,
    'rating': np.random.randint(1, 6, 20)
})

X, y = engineer_features(test_df)
#train a tiny ensemble (2 models, very few epochs for speed)
tiny_cfg = TrainConfig(epochs=5, batch_size=8, patience=3)
import torch
device = torch.device("cpu")
ensemble = fit_final_ensemble(X, y, n_models=2, device=device, cfg=tiny_cfg, seed=42)
test("Ensemble created with 2 models", len(ensemble.models) == 2)
test("Ensemble created with 2 scalers", len(ensemble.scalers) == 2)

#predict on the same data
preds = ensemble.predict(X)
test("Predictions have correct length", len(preds) == len(X))
test("All predictions between 1.0 and 5.0", preds.min() >= 1.0 and preds.max() <= 5.0,
     f"range: {preds.min():.2f}–{preds.max():.2f}")

#I3: Predictions -> Viewer pipeline (Obj 4.2)
print("\n--- I3: Predictions CSV -> Viewer load (Obj 4.2) ---")
tmp_pred_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w')
pred_df = pd.DataFrame({
    'track_name': ['Song A', 'Song B', 'Song C'],
    'track_artist': ['Artist 1', 'Artist 2', 'Artist 3'],
    'predicted_rating': [4.5, 3.2, 1.8]
})
pred_df.to_csv(tmp_pred_file.name, index=False)
tmp_pred_file.close()
loaded = load_predictions(tmp_pred_file.name)
test("Viewer loads predictions CSV", len(loaded) == 3)
test("predicted_rating column present", 'predicted_rating' in loaded.columns)
test("All ratings are floats", loaded['predicted_rating'].dtype == np.float64)
os.unlink(tmp_pred_file.name)

#summary
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)
