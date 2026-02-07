import pandas as pd
import tkinter as tk
from tkinter import messagebox
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# =========================
# Load dataset
# =========================
ratings = pd.read_csv("ratings.csv")


# =========================
# Create matrix
# =========================
def create_matrix(df):
    user_mapper = {uid: i for i, uid in enumerate(df['userId'].unique())}
    movie_mapper = {mid: i for i, mid in enumerate(df['movieId'].unique())}
    movie_inv_mapper = {i: mid for mid, i in movie_mapper.items()}

    user_index = df['userId'].map(user_mapper)
    movie_index = df['movieId'].map(movie_mapper)

    X = csr_matrix((df["rating"], (movie_index, user_index)),
                   shape=(len(movie_mapper), len(user_mapper)))

    return X, movie_mapper, movie_inv_mapper


X, movie_mapper, movie_inv_mapper = create_matrix(ratings)


# =========================
# Recommendation function
# =========================
def recommend_similar(movie_title, k=5):

    if movie_title not in ratings["title"].values:
        messagebox.showerror("Error", "Movie not found!")
        return

    movie_id = ratings[ratings['title'] == movie_title]['movieId'].iloc[0]
    movie_idx = movie_mapper[movie_id]
    movie_vec = X[movie_idx]

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(X)

    distances, indices = model.kneighbors(movie_vec, n_neighbors=k+1)

    neighbor_ids = [movie_inv_mapper[i] for i in indices.flatten()[1:]]
    recommendations = ratings[ratings['movieId'].isin(neighbor_ids)]['title'].unique()

    result_box.delete(0, tk.END)

    for movie in recommendations:
        result_box.insert(tk.END, movie)


# =========================
# Button action
# =========================
def run_recommend():
    movie = entry.get()
    k = int(num_entry.get())
    recommend_similar(movie, k)


# =========================
# GUI Design
# =========================
root = tk.Tk()
root.title("🎬 Movie Recommender")
root.geometry("400x400")

tk.Label(root, text="Enter Movie Name").pack(pady=5)
entry = tk.Entry(root, width=30)
entry.pack()

tk.Label(root, text="How many recommendations?").pack(pady=5)
num_entry = tk.Entry(root, width=5)
num_entry.insert(0, "5")
num_entry.pack()

tk.Button(root, text="Recommend", command=run_recommend).pack(pady=10)

result_box = tk.Listbox(root, width=40, height=10)
result_box.pack(pady=10)

root.mainloop()
