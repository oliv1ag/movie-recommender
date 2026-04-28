
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

# ----------------------------
# Load MovieLens data
# ----------------------------
def load_data(ratings_path, movies_path):
    ratings = pd.read_csv(ratings_path)
    ratings = ratings.sample(min(50000, len(ratings)))
    movies = pd.read_csv(movies_path)
    return ratings, movies

# ----------------------------
# Load TMDb data
# ----------------------------
def load_tmdb_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    return movies

# ----------------------------
# Helper functions
# ----------------------------
def extract_names(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def extract_top_cast(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except:
        return []

def extract_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name']
    except:
        return ""
    return ""

# ----------------------------
# Preprocess TMDb
# ----------------------------
def preprocess_tmdb(tmdb):
    tmdb['genres'] = tmdb['genres'].apply(extract_names)
    tmdb['keywords'] = tmdb['keywords'].apply(extract_names)
    tmdb['cast'] = tmdb['cast'].apply(extract_top_cast)
    tmdb['crew'] = tmdb['crew'].apply(extract_director)

    tmdb['overview'] = tmdb['overview'].fillna("")
    tmdb['original_language'] = tmdb['original_language'].fillna("")

    tmdb['tags'] = (
        tmdb['overview'] + " " +
        tmdb['genres'].astype(str) + " " +
        tmdb['keywords'].astype(str) + " " +
        tmdb['cast'].astype(str) + " " +
        tmdb['crew'].astype(str) + " " +
        tmdb['original_language'] + " " +
        tmdb['popularity'].astype(str)
    )

    return tmdb[['title', 'tags']]

# ----------------------------
# Merge + preprocess
# ----------------------------
def preprocess_data(ratings, movies):
    data = pd.merge(ratings, movies, on='movieId')

    # Load TMDb
    tmdb = load_tmdb_data()
    tmdb = preprocess_tmdb(tmdb)

    # Merge on title
    movies = movies.merge(tmdb, on='title', how='left')

    # Final text feature
    movies['overview'] = movies['tags'].fillna(movies['genres'])

    return data, movies

# ----------------------------
# Train/test split
# ----------------------------
def train_test_split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

