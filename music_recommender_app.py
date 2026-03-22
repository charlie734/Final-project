# main application window for the music recommendation system

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import threading

#import from existing modules
from song_rater import SongRaterApp
from train_nn_on_rated_songs import fit_model
from predict_top_songs import main as predict_main
from song_predictions_viewer import SongPredictionsViewer


class MusicRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Recommender System")
        self.root.geometry("600x400")
        self.root.resizable(True, True)

        #set up the main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        #title
        title_label = ttk.Label(
            self.main_frame, 
            text="Music Recommender System", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=20)

        #description
        description = (
            "Welcome to the Music Recommender System!\n\n"
            "This application allows you to rate songs, train a neural network model\n"
            "based on your ratings, and view personalized song recommendations."
        )
        desc_label = ttk.Label(
            self.main_frame, 
            text=description,
            justify=tk.CENTER
        )
        desc_label.pack(pady=10)

        #create buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.pack(pady=20)

        #create buttons
        ttk.Button(
            buttons_frame, 
            text="Rate Songs", 
            command=self.open_song_rater,
            width=20
        ).pack(pady=5)

        ttk.Button(
            buttons_frame, 
            text="Train Model", 
            command=self.train_model,
            width=20
        ).pack(pady=5)

        ttk.Button(
            buttons_frame, 
            text="View Predictions", 
            command=self.view_predictions,
            width=20
        ).pack(pady=5)

        ttk.Button(
            buttons_frame,
            text="Exit",
            command=self.root.destroy,
            width=20
        ).pack(pady=5)

        #status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        #check if the required files exist
        self.check_required_files()

    def check_required_files(self): #check if the required files exist and creates them if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))

        #check for spotify_songs.csv
        if not os.path.exists(os.path.join(current_dir, "spotify_songs.csv")):
            messagebox.showwarning(
                "Missing File", 
                "spotify_songs.csv not found. Some features may not work properly."
            )

        #check for ratings.csv - create if missing
        ratings_path = os.path.join(current_dir, "ratings.csv")
        if not os.path.exists(ratings_path):
            with open(ratings_path, "w") as f:
                f.write("track_id,rating\n")

        #create the models directory if it does not exist
        models_dir = os.path.join(current_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            self.status_var.set("Created models directory")

    def open_song_rater(self): #opens the song rater application
        self.status_var.set("Opening Song Rater...")
        self.root.update()

        #create a new window for the song rater
        rater_window = tk.Toplevel(self.root)
        rater_window.title("Song Rater")
        rater_window.geometry("1200x800")

        #initialise the song rater app
        song_rater = SongRaterApp(rater_window)
        self.status_var.set("Song Rater opened")

    def train_model(self): #train the neural network model based on the rated songs
        self.status_var.set("Preparing to train model...")
        self.root.update()

        #check if ratings exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ratings_path = os.path.join(current_dir, "ratings.csv")
        
        if not os.path.exists(ratings_path):
            messagebox.showerror(
                "Error", 
                "No ratings found. Please rate some songs first."
            )
            self.status_var.set("Training cancelled - no ratings found")
            return

        #check if there are enough ratings
        try:
            ratings_df = pd.read_csv(ratings_path)
            if len(ratings_df) < 10:
                result = messagebox.askyesno(
                    "Warning", 
                    f"You have only rated {len(ratings_df)} songs. "
                    f"It's recommended to rate at least 10 songs for better results. "
                    f"Continue anyway?"
                )
                if not result:
                    self.status_var.set("Training cancelled")
                    return
        except Exception as e:
            messagebox.showerror("Error", f"Could not read ratings file: {e}")
            return

        #show the progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Training Model")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        ttk.Label(
            progress_window,
            text="Training neural network model...\nThis may take several minutes.",
            justify=tk.CENTER
        ).pack(pady=20)

        progress = ttk.Progressbar(
            progress_window, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode='indeterminate'
        )
        progress.pack(pady=10)
        progress.start(10)

        #run training in a separate thread
        def train_thread():
            try:
                csv_path = os.path.join(current_dir, "rated_songs_with_features.csv")
                model_path = os.path.join(current_dir, "models", "best_cv_model.pkl")

                fit_model(
                    csv_path=csv_path,
                    device_pref="auto",
                    n_folds=5,
                    n_models=8,
                    save_best_to=model_path
                )

                #update UI on completion
                self.root.after(0, lambda: self.on_training_complete(progress_window))
            except Exception as e:
                error_message = str(e)
                self.root.after(0, lambda: self.on_training_error(progress_window, error_message))

        threading.Thread(target=train_thread, daemon=True).start()
        self.status_var.set("Training model...")

    def on_training_complete(self, progress_window): #called when training is complete
        progress_window.destroy()
        messagebox.showinfo(
            "Training Complete", 
            "The neural network model has been successfully trained!"
        )
        self.status_var.set("Model training completed")

    def on_training_error(self, progress_window, error_message): #callled when training encounters an error
        progress_window.destroy()
        messagebox.showerror(
            "Training Error", 
            f"An error occurred during model training:\n{error_message}"
        )
        self.status_var.set("Model training failed")

    def view_predictions(self): #view song predictions based on the trained model
        self.status_var.set("Preparing predictions...")
        self.root.update()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "best_cv_model.pkl")
        predictions_path = os.path.join(current_dir, "spotify_songs_with_predictions.csv")

        #check if the model exists
        if not os.path.exists(model_path):
            result = messagebox.askyesno(
                "Model Not Found", 
                "No trained model found. Would you like to train a model now?"
            )
            if result:
                self.train_model()
            else:
                self.status_var.set("Prediction cancelled - no model found")
            return

        #check if the prediction file exists
        if not os.path.exists(predictions_path):
            #generate predictions
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Generating Predictions")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()

            ttk.Label(
                progress_window,
                text="Generating song predictions...\nThis may take a few moments.",
                justify=tk.CENTER
            ).pack(pady=20)

            progress = ttk.Progressbar(
                progress_window, 
                orient=tk.HORIZONTAL, 
                length=300, 
                mode='indeterminate'
            )
            progress.pack(pady=10)
            progress.start(10)

            def predict_thread():
                try:
                    predict_main()
                    self.root.after(0, lambda: self.on_prediction_complete(progress_window))
                except Exception as e:
                    error_message = str(e)
                    self.root.after(0, lambda: self.on_prediction_error(progress_window, error_message))

            threading.Thread(target=predict_thread, daemon=True).start()
            self.status_var.set("Generating predictions...")
        else:
            #predictions already exist, open the viewer
            self.open_predictions_viewer()

    def on_prediction_complete(self, progress_window): #called when prediction generation is complete
        progress_window.destroy()
        self.open_predictions_viewer()
        self.status_var.set("Predictions generated")

    def on_prediction_error(self, progress_window, error_message): #called when prediction generation encounters an error
        progress_window.destroy()
        messagebox.showerror(
            "Prediction Error", 
            f"An error occurred during prediction generation:\n{error_message}"
        )
        self.status_var.set("Prediction generation failed")

    def open_predictions_viewer(self): #opens the song predictions viewer
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title("Song Predictions Viewer")
        viewer_window.geometry("1200x800")

        predictions_viewer = SongPredictionsViewer(viewer_window)
        self.status_var.set("Predictions viewer opened")


def main(): #the main function to run the application
    #set the working directory to the script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #create and run the application
    root = tk.Tk()
    app = MusicRecommenderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()