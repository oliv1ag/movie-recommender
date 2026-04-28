import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_indices = None

    def fit(self, movies_df):
        self.movie_indices = pd.Series(movies_df.index, index=movies_df['title'])

        self.tfidf_matrix = self.vectorizer.fit_transform(movies_df['overview'])

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, title, movies_df, top_k=10):
        idx = self.movie_indices[title]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]

        movie_indices = [i[0] for i in sim_scores]

        return movies_df['title'].iloc[movie_indices]