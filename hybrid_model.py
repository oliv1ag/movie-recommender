import numpy as np

class HybridModel:
    def __init__(self, content_model, collaborative_model, alpha=0.5):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha

    def hybrid_score(self, user_id, movie_idx, content_scores, movie_ids):
        content_score = content_scores[movie_idx]

        collab_score = self.collaborative_model.predict(
            user_id, movie_ids[movie_idx]
        )

        return self.alpha * content_score + (1 - self.alpha) * (collab_score / 5.0)

    def recommend(self, user_id, movies_df, content_scores, top_k=10):
        scores = []

        for i in range(len(movies_df)):
            score = self.hybrid_score(
                user_id, i, content_scores, movies_df['movieId'].values
            )
            scores.append((i, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

        indices = [i[0] for i in scores]
        return movies_df['title'].iloc[indices]