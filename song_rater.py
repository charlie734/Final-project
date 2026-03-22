#GUI for song rater

import csv
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd


def clean_data(): #clean the spotify songs by removing duplicates
    if not os.path.exists("../using 30000 song database/spotify_songs.csv"):
        print("spotify_songs.csv not found. Please ensure the dataset is available.")
        return False

    print("Cleaning data...")
    df = pd.read_csv("../using 30000 song database/spotify_songs.csv")
    
    #remove duplicates based on track_id
    original_count = len(df)
    df = df.drop_duplicates(subset=['track_id'], keep='first')
    cleaned_count = len(df)
    
    #save cleaned data
    df.to_csv("spotify_songs_clean.csv", index=False)
    
    print(f"Removed {original_count - cleaned_count} duplicates")
    print(f"Cleaned dataset saved with {cleaned_count} songs")
    return True


def load_data(): #load the cleaned data
    if not os.path.exists("../using 30000 song database/spotify_songs_clean.csv"):
        print("Cleaned dataset not found. Cleaning data first...")
        if not clean_data():
            return None
    
    try:
        df = pd.read_csv("../using 30000 song database/spotify_songs_clean.csv")
        print(f"Loaded {len(df)} songs from cleaned dataset")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_ratings(): #load existing ratings from the CSV file
    if not os.path.exists("ratings.csv"):
        return pd.DataFrame(columns=['track_id', 'rating'])
    
    try:
        return pd.read_csv("ratings.csv")
    except Exception as e:
        print(f"Error loading ratings: {e}")
        return pd.DataFrame(columns=['track_id', 'rating'])


def save_rating(track_id, rating): #save a new rating to the CSV file
    if not (1 <= rating <= 5):
        print(f"Invalid rating: {rating}. Must be between 1 and 5.")
        return False
    
    try:
        #load existing ratings
        ratings_df = load_ratings()
        
        #check if this track is already rated
        if track_id in ratings_df['track_id'].values:
            #update existing rating
            ratings_df.loc[ratings_df['track_id'] == track_id, 'rating'] = rating
            print(f"Updated rating for {track_id}: {rating}")
        else:
            #add a new rating
            new_rating = pd.DataFrame({'track_id': [track_id], 'rating': [rating]})
            ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
            print(f"Added new rating for {track_id}: {rating}")
        
        #save to CSV
        ratings_df.to_csv("ratings.csv", index=False)
        return True
        
    except Exception as e:
        print(f"Error saving rating: {e}")
        return False


class SongRaterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Song Rater")
        self.root.geometry("1000x700")
        
        #load data
        self.df = load_data()
        if self.df is None:
            messagebox.showerror("Error", "Could not load song data")
            return
        
        #create track_id to display mapping
        self.track_id_to_choice = {}
        self.choices = []
        
        for _, row in self.df.iterrows():
            choice = f"{row['track_name']} - {row['track_artist']}"
            self.choices.append(choice)
            self.track_id_to_choice[row['track_id']] = choice
        
        #variables
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_songs)
        self.track_id_var = tk.StringVar()
        self.rating_var = tk.IntVar()
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to rate songs")
        
        self.setup_ui()
        
        #load existing ratings count
        ratings_df = load_ratings()
        self.status_var.set(f"Ready - {len(ratings_df)} songs already rated")

    def setup_ui(self): #set up the user interface
        #main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        #title
        title_label = ttk.Label(main_frame, text="Song Rater", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        #instructions
        instructions = (
            "Rate songs to help train the recommendation system.\n"
            "Search for songs below and rate them from 1 (dislike) to 5 (love)."
        )
        ttk.Label(main_frame, text=instructions, justify=tk.CENTER).pack(pady=(0, 10))
        
        #search frame
        search_frame = ttk.LabelFrame(main_frame, text="Search Songs", padding="10")
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        search_entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        #song selection frame
        selection_frame = ttk.LabelFrame(main_frame, text="Select Song", padding="10")
        selection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        #song listbox with scrollbar
        listbox_frame = ttk.Frame(selection_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.song_listbox = tk.Listbox(listbox_frame, height=15)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.song_listbox.yview)
        self.song_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.song_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        #populate the listbox initially
        for choice in self.choices:
            self.song_listbox.insert(tk.END, choice)
        
        #bind selection event
        self.song_listbox.bind('<<ListboxSelect>>', self.on_song_select)
        
        #rating frame
        rating_frame = ttk.LabelFrame(main_frame, text="Rate Selected Song", padding="10")
        rating_frame.pack(fill=tk.X, pady=(0, 10))
        
        #selected song display
        ttk.Label(rating_frame, text="Selected:").pack(side=tk.LEFT)
        selected_label = ttk.Label(rating_frame, textvariable=self.track_id_var, width=50)
        selected_label.pack(side=tk.LEFT, padx=(5, 20))
        
        #rating buttons
        ttk.Label(rating_frame, text="Rating:").pack(side=tk.LEFT)
        for i in range(1, 6):
            ttk.Radiobutton(
                rating_frame, 
                text=str(i), 
                variable=self.rating_var, 
                value=i
            ).pack(side=tk.LEFT, padx=5)
        
        #control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Save Rating", command=self.save_rating).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="View Rated Songs", command=self.view_rated_songs).pack(side=tk.LEFT)
        
        #status bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def filter_songs(self, *args): #filter songs based on search text
        search_text = self.search_var.get().lower()
        
        #clear listbox
        self.song_listbox.delete(0, tk.END)
        
        #add matching songs
        for choice in self.choices:
            if search_text in choice.lower():
                self.song_listbox.insert(tk.END, choice)

    def on_song_select(self, event): #handle song selection from the listbox
        selection = self.song_listbox.curselection()
        if selection:
            choice = self.song_listbox.get(selection[0])
            
            #find the track_id for this choice
            for track_id, stored_choice in self.track_id_to_choice.items():
                if stored_choice == choice:
                    self.track_id_var.set(choice)
                    self.current_track_id = track_id
                    
                    #check if this song is already rated
                    ratings_df = load_ratings()
                    if track_id in ratings_df['track_id'].values:
                        existing_rating = ratings_df[ratings_df['track_id'] == track_id]['rating'].iloc[0]
                        self.rating_var.set(existing_rating)
                        self.status_var.set(f"Song already rated: {existing_rating}")
                    else:
                        self.rating_var.set(0)
                        self.status_var.set("Song selected - choose a rating")
                    break

    def save_rating(self): #save the current rating for the selected song
        if not hasattr(self, 'current_track_id'):
            messagebox.showwarning("No Selection", "Please select a song first")
            return
        
        if self.rating_var.get() == 0:
            messagebox.showwarning("No Rating", "Please select a rating (1-5)")
            return
        
        if save_rating(self.current_track_id, self.rating_var.get()):
            ratings_df = load_ratings()
            self.status_var.set(f"Rating saved! Total rated: {len(ratings_df)}")
            messagebox.showinfo("Success", "Rating saved successfully!")
        else:
            messagebox.showerror("Error", "Could not save rating")

    def clear_selection(self): #clear the selected song and rating
        self.song_listbox.selection_clear(0, tk.END)
        self.track_id_var.set("")
        self.rating_var.set(0)
        if hasattr(self, 'current_track_id'):
            delattr(self, 'current_track_id')
        self.status_var.set("Selection cleared")

    def view_rated_songs(self): #open a new window to view the rated songs
        ratings_df = load_ratings()
        
        if len(ratings_df) == 0:
            messagebox.showinfo("No Ratings", "You haven't rated any songs yet.")
            return
        
        #create a new window for rated songs
        rated_window = tk.Toplevel(self.root)
        rated_window.title("Your Rated Songs")
        rated_window.geometry("800x500")
        
        #main frame
        main_frame = ttk.Frame(rated_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        #title
        ttk.Label(main_frame, text=f"Your {len(ratings_df)} Rated Songs", 
                 font=("Helvetica", 14, "bold")).pack(pady=(0, 10))
        
        #create treeview
        columns = ("Song", "Artist", "Rating")
        tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=20)
        
        #define headings
        tree.heading("Song", text="Song")
        tree.heading("Artist", text="Artist")
        tree.heading("Rating", text="Rating")
        
        #define column widths
        tree.column("Song", width=400)
        tree.column("Artist", width=250)
        tree.column("Rating", width=100, anchor=tk.CENTER)
        
        #add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        #pack treeview and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        #populate treeview
        for _, row in ratings_df.iterrows():
            track_id = row['track_id']
            rating = row['rating']
            
            #get song info
            if track_id in self.track_id_to_choice:
                song_choice = self.track_id_to_choice[track_id]
                parts = song_choice.split(" - ")
                if len(parts) == 2:
                    song_name, artist = parts
                else:
                    song_name, artist = song_choice, "Unknown"
            else:
                song_name, artist = "Unknown", "Unknown"
            
            tree.insert("", tk.END, values=(song_name, artist, rating))
        
        #close button
        ttk.Button(main_frame, text="Close", command=rated_window.destroy).pack(pady=10)


def main(): #main function to run the application
    root = tk.Tk()
    app = SongRaterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()