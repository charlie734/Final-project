#merges the ratings with song features from the Spotify datasets

import os
import pandas as pd


def load_and_clean_dataset(file_path, dataset_name):
    """
    load a dataset and perform cleaning.

    arguments:
        file_path: Path to the CSV file
        dataset_name: Name for logging purposes

    returns:
        pd.DataFrame: Cleaned dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{dataset_name} file not found: {file_path}")

    print(f"Loading {dataset_name} from {file_path}")
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty")

    #remove duplicates if there are any
    initial_size = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_size:
        print(f"Removed {initial_size - len(df)} duplicates from {dataset_name}")

    print(f"{dataset_name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_datasets(
    ratings_file="ratings.csv",
    songs_file="spotify_songs.csv",
    output_file="rated_songs_with_features.csv"
):
    """
    merge the ratings with song features from Spotify datasets.

    arguments:
        ratings_file: Path to ratings CSV
        songs_file: Path to songs CSV
        output_file: Path for merged output

    returns:
        pd.DataFrame: Merged dataset
    """
    print("Starting dataset merge...")

    #load datasets
    print("\n=== Loading Datasets ===")
    ratings_df = load_and_clean_dataset(ratings_file, "ratings")
    songs_df = load_and_clean_dataset(songs_file, "songs")

    #check that required columns exist
    if 'track_id' not in ratings_df.columns:
        raise ValueError("ratings dataset missing 'track_id' column")
    if 'rating' not in ratings_df.columns:
        raise ValueError("ratings dataset missing 'rating' column")
    if 'track_id' not in songs_df.columns:
        raise ValueError("songs dataset missing 'track_id' column")

    #merge ratings with songs (inner join to keep only rated songs)
    print("\n=== Merging ratings with songs ===")
    final_df = pd.merge(ratings_df, songs_df, on='track_id', how='inner')
    print(f"After merge: {len(final_df)} rows")

    #check for missing features
    feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
    available_features = [col for col in feature_columns if col in final_df.columns]

    if available_features:
        missing_features = final_df[available_features].isna().all(axis=1).sum()
        if missing_features > 0:
            print(f"Warning: {missing_features} songs have missing audio features")

    #save the merged dataset
    print(f"\n=== Saving merged dataset ===")
    final_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to: {output_file}")
    print(f"Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")

    return final_df


def validate_merged_dataset(file_path="rated_songs_with_features.csv"):
    """
    validation of the merged dataset.

    arguments:
        file_path: Path to the merged dataset

    returns:
        bool: True if validation passes
    """
    if not os.path.exists(file_path):
        print(f"Error: Merged dataset not found at {file_path}")
        return False

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            print("Error: Merged dataset is empty")
            return False

        #check for required columns
        required_columns = ['track_id', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False

        print(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
        print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")

        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def main(): #main function to merge datasets
    try:
        #merge the datasets
        merged_df = merge_datasets()

        #validate the result
        if validate_merged_dataset():
            print("\nDataset merge completed successfully!")
        else:
            print("\nDataset merge completed but validation failed!")

    except Exception as e:
        print(f"Error during merge: {e}")


if __name__ == "__main__":
    main()
