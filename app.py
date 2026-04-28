
import streamlit as st
import pandas as pd

from data_loader import load_data, preprocess_data, train_test_split_data
from content_model import ContentModel
from collaborative_model import CollaborativeModel
from hybrid_model import HybridModel

# ----------------------------
# Load Data (cached)
# ----------------------------
@st.cache_data
def load_all():
    ratings, movies = load_data("ratings_small.csv", "movies_small.csv")
    data, movies = preprocess_data(ratings, movies)
    train, test = train_test_split_data(data)
    return ratings, movies, train, test

ratings, movies, train, test = load_all()

# ----------------------------
# Train Models (cached)
# ----------------------------
@st.cache_resource
def train_models():
    content_model = ContentModel()
    content_model.fit(movies)

    collab_model = CollaborativeModel()
    collab_model.fit(ratings)

    return content_model, collab_model

content_model, collab_model = train_models()

# ----------------------------
# UI
# ----------------------------
st.title("🎬 Hybrid Movie Recommender (TMDb + MovieLens)")

st.markdown("Combines **content-based (NLP on metadata)** + **collaborative filtering (SVD)**")

user_id = st.selectbox("Select User ID", sorted(ratings['userId'].unique()))

movie_title = st.selectbox("Select a Movie You Like", movies['title'].values)

alpha = st.slider("Hybrid Weight (alpha)", 0.0, 1.0, 0.7)

top_k = st.slider("Number of Recommendations", 5, 20, 10)

# ----------------------------
# Recommendation Function
# ----------------------------
def get_collab_recommendations(model, user_id, movies_df, train_df, top_k=10):
    scores = []
    seen_movies = train_df[train_df['userId'] == user_id]['movieId'].tolist()

    for movie_id in movies_df['movieId']:
        if movie_id in seen_movies:
            continue

        pred = model.predict(user_id, movie_id)
        scores.append((movie_id, pred))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    movie_ids = [x[0] for x in scores]

    return movies_df[movies_df['movieId'].isin(movie_ids)]['title'].tolist()

# ----------------------------
# Button Action
# ----------------------------
if st.button("🎯 Get Recommendations"):

    hybrid_model = HybridModel(content_model, collab_model, alpha=alpha)

    # Content-based
    content_recs = content_model.recommend(movie_title, movies, top_k)

    # Collaborative
    collab_recs = get_collab_recommendations(collab_model, user_id, movies, train, top_k)

    # Hybrid
    movie_idx = movies[movies['title'] == movie_title].index[0]

    hybrid_recs = hybrid_model.recommend(
        user_id,
        movies,
        content_model.similarity_matrix[movie_idx],
        top_k
    )

    # ----------------------------
    # DISPLAY RESULTS
    # ----------------------------
    st.subheader("🎯 Content-Based Recommendations")
    for i, m in enumerate(content_recs, 1):
        st.write(f"{i}. {m}")

    st.subheader("🤝 Collaborative Recommendations")
    for i, m in enumerate(collab_recs, 1):
        st.write(f"{i}. {m}")


    st.subheader("🔥 Hybrid Recommendations (Best)")

    for i, m in enumerate(hybrid_recs, 1):
        st.write(f"{i}. {m}")

    st.info(
        f"""
    These recommendations combine:

    • Content similarity (genre, cast, keywords, etc.)
    • Collaborative filtering (user rating patterns)

    Hybrid score = {alpha:.2f} × Content + {1-alpha:.2f} × Collaborative
    """
    )



    # ----------------------------
    # INFO PANEL
    # ----------------------------
    st.info(f"""
    Hybrid score = {alpha:.2f} × Content + {1-alpha:.2f} × Collaborative
    
    ✔ Content uses: genres, cast, crew, keywords, language, popularity  
    ✔ Collaborative uses: user rating patterns (SVD)  
    ✔ Hybrid combines both for better recommendations  
    """)

