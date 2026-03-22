#loads the model and predicts ratings for all songs in the dataset

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

#import the necessary functions and classes from train_nn_on_rated_songs.py
from train_nn_on_rated_songs import (
    RatingMLP, 
    load_model_from_pickle, 
    engineer_features, 
    get_device
)


def load_spotify_songs(csv_path: str = "spotify_songs.csv") -> pd.DataFrame:
    """
    load the Spotify songs dataset.

    arguments:
        csv_path: Path to the CSV file containing song data

    returns:
        DataFrame containing the songs
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Spotify songs file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Spotify songs dataset is empty")

    print(f"Loaded {len(df)} songs from {csv_path}")
    return df


def predict_ratings(model: nn.Module, bundle: Dict, songs_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """
    predict ratings for all songs using the trained model.

    arguments:
        model: Trained PyTorch model
        bundle: Dictionary containing model metadata (scaler, feature columns)
        songs_df: DataFrame containing the songs to predict ratings for
        device: PyTorch device to use for prediction

    returns:
        DataFrame with track_id, track_name, track_artist, and predicted_rating columns
    """
    #add a placeholder rating column for engineer_features to work
    songs_df = songs_df.copy()
    songs_df["rating"] = 0.0

    #engineer features
    X_df, _ = engineer_features(songs_df)

    #get the scaler from the bundle
    scaler = bundle.get("scaler")
    if scaler is None:
        raise ValueError("Model bundle does not contain a scaler")

    #check for feature columns mismatch
    expected_features = bundle.get("feature_columns")
    if expected_features is not None:
        #make sure columns are in the same order as expected by the model
        X_df = X_df[expected_features]

    #scale the features
    X_np = X_df.to_numpy().astype(np.float32)
    X_scaled = scaler.transform(X_np)

    #convert to tensor and predict
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
        #clamp predictions to valid rating range
        predictions = np.clip(predictions, 1.0, 5.0)

    #create results dataframe
    result_df = pd.DataFrame({
        "track_id": songs_df["track_id"],
        "track_name": songs_df["track_name"],
        "track_artist": songs_df["track_artist"],
        "predicted_rating": predictions
    })

    return result_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove duplicate songs, keeping the highest rated version.

    arguments:
        df: DataFrame with song predictions

    returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df

    #deduplication based on track name and artist
    df_dedup = df.sort_values('predicted_rating', ascending=False).drop_duplicates(
        subset=['track_name', 'track_artist'], keep='first'
    ).reset_index(drop=True)

    num_removed = len(df) - len(df_dedup)
    if num_removed > 0:
        print(f"Removed {num_removed} duplicate songs")

    return df_dedup


def get_top_songs(predictions_df: pd.DataFrame, n: int = 50, remove_dups: bool = True) -> pd.DataFrame:
    """
    get the top N songs with the highest predicted ratings.

    arguments:
        predictions_df: DataFrame with predicted ratings
        n: Number of top songs to return
        remove_dups: Whether to remove duplicate songs

    returns:
        DataFrame with the top N songs
    """
    if predictions_df.empty:
        raise ValueError("No predictions available")

    df = predictions_df.copy()

    #sort by predicted rating (highest first)
    df = df.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

    #remove duplicates if requested
    if remove_dups:
        df = remove_duplicates(df)

    #return top N songs
    return df.head(n)


def display_top_songs(top_songs: pd.DataFrame, n: Optional[int] = None):
    """
    display the top songs in a formatted way.

    arguments:
        top_songs: DataFrame containing the top songs
        n: Number of songs to display (if None, display all)
    """
    if top_songs.empty:
        print("No songs to display")
        return

    if n is not None:
        display_df = top_songs.head(n)
    else:
        display_df = top_songs

    max_rating = top_songs["predicted_rating"].max()
    print(f"\nMaximum predicted rating: {max_rating:.4f}")

    print(f"\nTop {len(display_df)} songs with highest predicted ratings:")
    print(display_df[["track_name", "track_artist", "predicted_rating"]].to_string(index=False))


def save_predictions_to_csv(songs_df: pd.DataFrame, predictions_df: pd.DataFrame, 
                           output_path: str = "spotify_songs_with_predictions.csv") -> pd.DataFrame:
    """
    save all songs with their predicted ratings to a CSV file.

    arguments:
        songs_df: Original DataFrame containing all song data
        predictions_df: DataFrame with track_id and predicted_rating columns
        output_path: Path to save the CSV file

    returns:
        DataFrame containing all song data with predicted ratings
    """
    #merge the original songs dataframe with the predictions
    merged_df = songs_df.copy()

    #create a mapping from track_id to predicted_rating
    predictions_map = dict(zip(predictions_df["track_id"], predictions_df["predicted_rating"]))

    #add the predicted ratings to the merged dataframe
    merged_df["predicted_rating"] = merged_df["track_id"].map(predictions_map)

    #sort by predicted rating in descending order
    merged_df = merged_df.sort_values("predicted_rating", ascending=False)

    #remove duplicates, keeping the highest rated version
    original_count = len(merged_df)
    merged_df = remove_duplicates(merged_df)
    removed_count = original_count - len(merged_df)

    if removed_count > 0:
        print(f"Removed {removed_count} duplicate songs from predictions")

    #save to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"Saved {len(merged_df)} songs with predictions to {output_path}")

    return merged_df


def main(): #main function to run the program
    try:
        #set device to auto-detect GPU or CPU
        device = get_device("auto")
        print(f"Using device: {device}")

        #load the trained model
        model_path = "../using 1.2 million song database/models/best_cv_model.pkl"
        print(f"Loading model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")

        model, bundle = load_model_from_pickle(model_path)
        model.to(device)
        model.eval()

        #load Spotify songs
        print("Loading Spotify songs dataset")
        songs_df = load_spotify_songs()

        #check for required columns
        required_columns = ["track_id", "track_name", "track_artist"]
        missing_columns = [col for col in required_columns if col not in songs_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        #predict ratings
        print("Predicting ratings for all songs")
        predictions_df = predict_ratings(model, bundle, songs_df, device)
        print(f"Generated predictions for {len(predictions_df)} songs")

        #verify predictions range
        min_pred = predictions_df["predicted_rating"].min()
        max_pred = predictions_df["predicted_rating"].max()
        print(f"Prediction range: {min_pred:.4f} to {max_pred:.4f}")

        #save all songs with their predicted ratings to a CSV file
        print("\nSaving all songs with predictions to CSV...")
        songs_with_predictions = save_predictions_to_csv(songs_df, predictions_df)

        #get and display the top 50 songs
        print("\nGetting top songs...")
        top_songs = get_top_songs(predictions_df, n=50, remove_dups=True)
        display_top_songs(top_songs)

        return top_songs, songs_with_predictions

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()
