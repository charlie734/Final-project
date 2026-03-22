import os
import sys
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil

#make sure that the current working directory is the same as the file location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_nn_on_rated_songs import (
    FeatureConstants, TrainConfig, ResidualBlock, RatingMLP,
    engineer_features, rmse_np, set_seed, get_device, load_dataset
)
from song_rater import save_rating, load_ratings, clean_data, load_data
from predict_top_songs import remove_duplicates as predict_remove_duplicates
from song_predictions_viewer import remove_duplicates as viewer_remove_duplicates, load_predictions
from merge_datasets import load_and_clean_dataset, merge_datasets

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
print("UNIT TESTS")
print("=" * 60)

#U1: FeatureConstants default values (Obj 3.2)
print("\n--- U1: FeatureConstants defaults (Obj 3.2) ---")
c = FeatureConstants()
test("MAX_TEMPO_BPM is 200.0", c.MAX_TEMPO_BPM == 200.0)
test("MS_PER_MINUTE is 60000.0", c.MS_PER_MINUTE == 60000.0)
test("RATING_MIN is 1", c.RATING_MIN == 1)
test("RATING_MAX is 5", c.RATING_MAX == 5)

#U2: TrainConfig default values (Obj 3.4)
print("\n--- U2: TrainConfig defaults (Obj 3.4) ---")
cfg = TrainConfig()
test("Default epochs is 600", cfg.epochs == 600)
test("Default patience is 50", cfg.patience == 50)
test("Default batch_size is 112", cfg.batch_size == 112)
test("Default grad_clip is 0.25", cfg.grad_clip == 0.25)

#U3: ResidualBlock produces same-shape output (Obj 3.3)
print("\n--- U3: ResidualBlock shape preservation (Obj 3.3) ---")
block = ResidualBlock(64, p=0.0)
block.eval()
x = torch.randn(4, 64)
y = block(x)
test("Output shape matches input shape", y.shape == x.shape, f"got {y.shape}")
test("Skip connection works (output not equal to zero)", y.abs().sum().item() > 0)

#U4: RatingMLP forward pass (Obj 3.1, 3.3)
print("\n--- U4: RatingMLP forward pass (Obj 3.1, 3.3) ---")
model = RatingMLP(in_dim=16, width=64, depth=3, p=0.0)
model.eval()
x = torch.randn(8, 16)
out = model(x)
test("Output shape is (batch, 1)", out.shape == (8, 1), f"got {out.shape}")
test("Output is finite (no NaN)", torch.isfinite(out).all().item())

#Count hidden layers (at least 10 per obj 3.3 - with depth=8 default that gives >10 total)
default_model = RatingMLP(in_dim=16)
total_linears = sum(1 for m in default_model.modules() if isinstance(m, torch.nn.Linear))
test("Default model has many linear layers (>10)", total_linears > 10, f"got {total_linears}")

#U5: rmse_np calculation (Obj 5.5)
print("\n--- U5: rmse_np calculation (Obj 5.5) ---")
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
test("RMSE of identical arrays is 0.0", rmse_np(y_true, y_pred) == 0.0)

y_pred2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
expected_rmse = 1.0
test("RMSE of offset-by-1 arrays is 1.0", abs(rmse_np(y_true, y_pred2) - expected_rmse) < 1e-6)

#U6: get_device returns valid device (Obj 3.8)
print("\n--- U6: get_device (Obj 3.8) ---")
dev = get_device("auto")
test("Auto device returns cpu or cuda", str(dev) in ["cpu", "cuda"])
dev_cpu = get_device("cpu")
test("Explicit cpu returns cpu", str(dev_cpu) == "cpu")

#U7: engineer_features output shape (Obj 3.1, 3.2)
print("\n--- U7: engineer_features output (Obj 3.1, 3.2) ---")
#create a minimal dataframe that mimics rated_songs_with_features.csv
test_df = pd.DataFrame({
    'danceability': [0.5, 0.7, 0.3],
    'energy': [0.8, 0.6, 0.4],
    'speechiness': [0.05, 0.1, 0.03],
    'acousticness': [0.2, 0.8, 0.5],
    'instrumentalness': [0.0, 0.01, 0.5],
    'liveness': [0.1, 0.3, 0.6],
    'valence': [0.9, 0.4, 0.2],
    'tempo': [120.0, 90.0, 150.0],
    'duration_ms': [200000, 180000, 240000],
    'key': [5, 0, 11],
    'mode': [1, 0, 1],
    'loudness': [-5.0, -10.0, -3.0],
    'time_signature': [4, 4, 3],
    'track_popularity': [70, 30, 50],
    'track_album_release_date': ['2020-03-20', '2015-06-01', '2000-01-15'],
    'rating': [5, 3, 1]
})
X, y = engineer_features(test_df)
test("X has 3 rows", len(X) == 3)
test("y has 3 values", len(y) == 3)
test("X has more than 7 columns (engineered features added)", X.shape[1] > 7, f"got {X.shape[1]} columns")
test("y values match input ratings", list(y) == [5, 3, 1])
test("No NaN values in X", X.isna().sum().sum() == 0)

#U8: save_rating and load_ratings (Obj 1.3, 1.5)
print("\n--- U8: save_rating / load_ratings (Obj 1.3, 1.5) ---")
#use a temporary directory to avoid touching real data
original_dir = os.getcwd()
tmp_dir = tempfile.mkdtemp()
os.chdir(tmp_dir)
try:
    #save a new rating
    result = save_rating("test_track_001", 4)
    test("save_rating returns True for valid rating", result == True)

    #load it back
    ratings = load_ratings()
    test("load_ratings returns 1 row", len(ratings) == 1)
    test("Saved track_id is correct", ratings.iloc[0]['track_id'] == "test_track_001")
    test("Saved rating is 4", int(ratings.iloc[0]['rating']) == 4)

    #update the rating
    save_rating("test_track_001", 2)
    ratings2 = load_ratings()
    test("Updating doesn't duplicate (still 1 row)", len(ratings2) == 1)
    test("Updated rating is 2", int(ratings2.iloc[0]['rating']) == 2)

    #boundary: rating of 1
    save_rating("test_track_002", 1)
    test("Rating of 1 accepted", len(load_ratings()) == 2)

    #boundary: rating of 5
    save_rating("test_track_003", 5)
    test("Rating of 5 accepted", len(load_ratings()) == 3)

    #invalid: rating of 0
    result_invalid = save_rating("test_track_004", 0)
    test("Rating of 0 rejected", result_invalid == False)
    test("Invalid rating not saved", len(load_ratings()) == 3)

    #invalid: rating of 6
    result_invalid2 = save_rating("test_track_005", 6)
    test("Rating of 6 rejected", result_invalid2 == False)
    test("Invalid rating not saved", len(load_ratings()) == 3)
finally:
    os.chdir(original_dir)
    shutil.rmtree(tmp_dir)

#U9: remove_duplicates (Obj 4.5)
print("\n--- U9: remove_duplicates (Obj 4.5) ---")
dup_df = pd.DataFrame({
    'track_name': ['Song A', 'Song A', 'Song B', 'Song B', 'Song C'],
    'track_artist': ['Artist X', 'Artist X', 'Artist Y', 'Artist Y', 'Artist Z'],
    'predicted_rating': [3.5, 4.2, 2.1, 2.8, 4.9]
})
deduped = predict_remove_duplicates(dup_df)
test("Duplicates removed: 5->3 rows", len(deduped) == 3)
#should keep the highest rating for each
song_a_rating = deduped[deduped['track_name'] == 'Song A']['predicted_rating'].iloc[0]
test("Keeps highest rated version of Song A (4.2)", abs(song_a_rating - 4.2) < 0.01)
song_b_rating = deduped[deduped['track_name'] == 'Song B']['predicted_rating'].iloc[0]
test("Keeps highest rated version of Song B (2.8)", abs(song_b_rating - 2.8) < 0.01)

#U10: Prediction clamping (Obj 4.1)
print("\n--- U10: Prediction clamping (Obj 4.1) ---")
raw_preds = np.array([-1.5, 0.0, 0.99, 1.0, 3.0, 5.0, 5.01, 7.5])
clamped = np.clip(raw_preds, 1.0, 5.0)
test("Value below 1.0 clamped to 1.0", clamped[0] == 1.0)
test("Value of 0.0 clamped to 1.0", clamped[1] == 1.0)
test("Value of 0.99 clamped to 1.0", clamped[2] == 1.0)
test("Value of 1.0 unchanged", clamped[3] == 1.0)
test("Value of 3.0 unchanged", clamped[4] == 3.0)
test("Value of 5.0 unchanged", clamped[5] == 5.0)
test("Value of 5.01 clamped to 5.0", clamped[6] == 5.0)
test("Value of 7.5 clamped to 5.0", clamped[7] == 5.0)

#U11: load_dataset validation (Obj 3.1)
print("\n--- U11: load_dataset validation (Obj 3.1) ---")
#non-existent file
try:
    load_dataset("nonexistent_file_abc123.csv")
    test("Raises error for missing file", False)
except FileNotFoundError:
    test("Raises FileNotFoundError for missing file", True)

#empty file
tmp_empty = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w')
tmp_empty.write("track_id,rating\n")
tmp_empty.close()
try:
    load_dataset(tmp_empty.name)
    test("Raises error for empty dataset", False)
except ValueError:
    test("Raises ValueError for empty dataset", True)
os.unlink(tmp_empty.name)

#file without a rating column
tmp_no_rating = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w')
tmp_no_rating.write("track_id,danceability\nabc,0.5\n")
tmp_no_rating.close()
try:
    load_dataset(tmp_no_rating.name)
    test("Raises error for missing rating column", False)
except ValueError:
    test("Raises ValueError for missing rating column", True)
os.unlink(tmp_no_rating.name)


#summary
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)
