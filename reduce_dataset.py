id="q4bbd9"
import pandas as pd

print("Reducing MovieLens dataset...")

# Load original datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# -----------------------------
# REDUCE RATINGS
# -----------------------------
ratings_small = ratings.sample(n=20000, random_state=42)

# Keep only movies that appear in sampled ratings
valid_movie_ids = ratings_small['movieId'].unique()

movies_small = movies[movies['movieId'].isin(valid_movie_ids)]

# -----------------------------
# SAVE SMALL FILES
# -----------------------------
ratings_small.to_csv("ratings_small.csv", index=False)
movies_small.to_csv("movies_small.csv", index=False)

print("Done!")
print("ratings_small:", len(ratings_small))
print("movies_small:", len(movies_small))

