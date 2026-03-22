#downloads the 30000-song dataset from kaggle

import os
import kagglehub
import pandas as pd


def download_spotify_dataset(dataset_name="joebeachcapital/30000-spotify-songs", target_dir="spotify_data"):
    """
    download the Spotify dataset from Kaggle.
    
    arguments:
        dataset_name: Kaggle dataset identifier
        target_dir: Directory to store the dataset
        
    returns:
        str: Path to the downloaded dataset
    """
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        #download dataset using kagglehub
        dataset_path = kagglehub.dataset_download(dataset_name)
        
        if not dataset_path or not os.path.exists(dataset_path):
            raise Exception("Download failed - invalid path")
        
        print(f"Dataset downloaded to: {dataset_path}")
        
        #validation: check if the main file exists
        main_file = os.path.join(dataset_path, "spotify_songs.csv")
        if os.path.exists(main_file):
            #check that the file is readable
            df = pd.read_csv(main_file, nrows=5)
            print(f"Dataset validated - {len(df.columns)} columns found")
        else:
            print("Warning: Main CSV file not found, but download completed")
        
        return dataset_path
        
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def validate_dataset(dataset_path):
    """
    validation of the downloaded dataset.
    
    arguments:
        dataset_path: Path to the dataset directory
        
    returns:
        bool: True if validation passes
    """
    main_file = os.path.join(dataset_path, "spotify_songs.csv")
    
    if not os.path.exists(main_file):
        print("Error: spotify_songs.csv not found")
        return False
    
    if os.path.getsize(main_file) == 0:
        print("Error: spotify_songs.csv is empty")
        return False
    
    try:
        # Try to read the file
        df = pd.read_csv(main_file, nrows=10)
        print(f"Validation passed - {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def main(): #the main function to download the dataset
    try:
        # Download the dataset
        dataset_path = download_spotify_dataset()
        
        # Validate the dataset
        if validate_dataset(dataset_path):
            print("Dataset download and validation completed successfully!")
            print(f"Dataset location: {dataset_path}")
        else:
            print("Dataset validation failed!")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()