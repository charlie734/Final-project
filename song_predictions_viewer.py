#prediction viewer to sort and filter song predictions

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from typing import Optional


def load_predictions(csv_path: str = "spotify_songs_with_predictions.csv") -> pd.DataFrame:
    """
    load song predictions from the CSV file.

    arguments:
        csv_path: Path to the CSV file containing predictions

    returns:
        DataFrame containing song predictions
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Predictions file is empty")

    if 'predicted_rating' not in df.columns:
        raise ValueError("Predictions file missing 'predicted_rating' column")

    print(f"Loaded {len(df)} song predictions")
    return df


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

    return df_dedup


class SongPredictionsViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Song Predictions Viewer")
        self.root.geometry("1200x800")

        #load predictions data
        try:
            self.df = load_predictions()
            #automatically remove duplicates when loading data
            original_count = len(self.df)
            self.df = remove_duplicates(self.df)
            removed_count = original_count - len(self.df)
            if removed_count > 0:
                print(f"Automatically removed {removed_count} duplicate songs from predictions")
            self.filtered_df = self.df.copy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load predictions: {e}")
            self.root.destroy()
            return

        #create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        #title
        title_label = ttk.Label(
            self.main_frame, 
            text="Song Predictions Viewer", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        #info label
        self.info_label = ttk.Label(
            self.main_frame,
            text=f"Showing {len(self.filtered_df)} songs (sorted by predicted rating)"
        )
        self.info_label.pack(pady=(0, 10))

        #control frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        #filter controls
        filter_frame = ttk.LabelFrame(control_frame, text="Filters", padding="5")
        filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        #minimum rating filter
        ttk.Label(filter_frame, text="Min Rating:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_rating_var = tk.StringVar(value="1.0")
        min_rating_spinbox = ttk.Spinbox(
            filter_frame, 
            from_=1.0, 
            to=5.0, 
            increment=0.1, 
            width=10,
            textvariable=self.min_rating_var,
            command=self.apply_filters
        )
        min_rating_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        #search filter
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.apply_filters)
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=(0, 10))

        #reset button
        ttk.Button(filter_frame, text="Reset", command=self.reset_filters).pack(side=tk.LEFT)

        #action buttons frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)

        ttk.Button(
            button_frame, 
            text="Remove Duplicates", 
            command=self.remove_duplicates
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            button_frame, 
            text="Export Top 100", 
            command=self.export_top_songs
        ).pack(side=tk.LEFT)

        #create a treeview for songs
        self.create_song_treeview()

    def create_song_treeview(self): #create a treeview to display songs
        #create a frame for treeview and scrollbar
        tree_frame = ttk.Frame(self.main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        #define columns
        columns = ("Track Name", "Artist", "Predicted Rating")

        #create treeview
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)

        #define column headings and widths
        self.tree.heading("Track Name", text="Track Name")
        self.tree.heading("Artist", text="Artist")
        self.tree.heading("Predicted Rating", text="Predicted Rating")

        self.tree.column("Track Name", width=400)
        self.tree.column("Artist", width=300)
        self.tree.column("Predicted Rating", width=150)

        #add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)

        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        #pack treeview and scrollbars
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        #populate treeview
        self.populate_treeview()

    def populate_treeview(self): #populate the treeview with song data
        #clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        #add songs to treeview (limit to top 1000 for performance)
        display_df = self.filtered_df.head(1000)

        for _, row in display_df.iterrows():
            track_name = str(row.get('track_name', 'Unknown'))
            artist = str(row.get('track_artist', 'Unknown'))
            rating = f"{row['predicted_rating']:.3f}"

            self.tree.insert("", tk.END, values=(track_name, artist, rating))

        #update info label
        total_songs = len(self.filtered_df)
        displayed_songs = len(display_df)

        if displayed_songs < total_songs:
            info_text = f"Showing top {displayed_songs} of {total_songs} songs"
        else:
            info_text = f"Showing {displayed_songs} songs"

        self.info_label.config(text=info_text)

    def apply_filters(self, *args): #apply filters to the song data
        try:
            #start with the full dataset
            filtered_df = self.df.copy()

            #apply minimum rating filter
            min_rating = float(self.min_rating_var.get())
            filtered_df = filtered_df[filtered_df['predicted_rating'] >= min_rating]

            #apply search filter
            search_text = self.search_var.get().strip().lower()
            if search_text:
                mask = (
                    filtered_df['track_name'].str.lower().str.contains(search_text, na=False) |
                    filtered_df['track_artist'].str.lower().str.contains(search_text, na=False)
                )
                filtered_df = filtered_df[mask]

            #sort by predicted rating (highest first)
            filtered_df = filtered_df.sort_values('predicted_rating', ascending=False)

            self.filtered_df = filtered_df
            self.populate_treeview()

        except ValueError:
            #handle invalid rating input
            pass

    def reset_filters(self): #reset all filters to default values
        self.min_rating_var.set("1.0")
        self.search_var.set("")
        self.filtered_df = self.df.copy()
        self.populate_treeview()

    def remove_duplicates(self): #remove duplicate songs from the filtered data
        try:
            original_count = len(self.filtered_df)
            self.filtered_df = remove_duplicates(self.filtered_df)
            removed_count = original_count - len(self.filtered_df)

            self.populate_treeview()

            if removed_count > 0:
                messagebox.showinfo("Duplicates Removed", f"Removed {removed_count} duplicate songs")
            else:
                messagebox.showinfo("No Duplicates", "No duplicate songs found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove duplicates: {e}")

    def export_top_songs(self): #export top 100 songs to a CSV file
        try:
            top_songs = self.filtered_df.head(100)
            output_path = "top_100_predictions.csv"

            # Select relevant columns for export
            export_columns = ['track_name', 'track_artist', 'predicted_rating']
            available_columns = [col for col in export_columns if col in top_songs.columns]

            top_songs[available_columns].to_csv(output_path, index=False)

            messagebox.showinfo(
                "Export Complete", 
                f"Top {len(top_songs)} songs exported to {output_path}"
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export songs: {e}")


def main(): #main function to create and run the GUI
    root = tk.Tk()
    app = SongPredictionsViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
