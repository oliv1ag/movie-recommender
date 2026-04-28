
print("STARTING PROJECT...")

import matplotlib.pyplot as plt

from data_loader import load_data, preprocess_data, train_test_split_data
from content_model import ContentModel
from collaborative_model import CollaborativeModel
from hybrid_model import HybridModel
from evaluation import evaluate_models, precision_at_k, recall_at_k


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
    print("Using ratings from:", len(train_df))


# ----------------------------
# LOAD DATA
# ----------------------------
ratings, movies = load_data("ratings_small.csv", "movies_small.csv")
data, movies = preprocess_data(ratings, movies)

train, test = train_test_split_data(data)

# ----------------------------
# CONTENT MODEL
# ----------------------------
content_model = ContentModel()
content_model.fit(movies)

# ----------------------------
# COLLAB MODEL
# ----------------------------
collab_model = CollaborativeModel()
collab_model.fit(ratings)

rmse_score = collab_model.evaluate()
print("RMSE:", rmse_score)

# ----------------------------
# HYBRID MODEL
# ----------------------------
hybrid_model = HybridModel(content_model, collab_model, alpha=0.7)

# ----------------------------
# SAMPLE USER
# ----------------------------
user_id = 1
user_test_data = test[test['userId'] == user_id]

ground_truth = user_test_data[user_test_data['rating'] >= 4]['title'].tolist()

if len(ground_truth) == 0:
    ground_truth = movies['title'].iloc[:10].tolist()

liked_movie = ground_truth[0]

# ----------------------------
# RECOMMENDATIONS
# ----------------------------
content_recs = content_model.recommend(liked_movie, movies)
collab_recs = get_collab_recommendations(collab_model, user_id, movies, train)

movie_idx = movies[movies['title'] == liked_movie].index[0]

hybrid_recs = hybrid_model.recommend(
    user_id,
    movies,
    content_model.similarity_matrix[movie_idx]
)

# ----------------------------
# OUTPUT
# ----------------------------
print("\nGround Truth:", ground_truth[:5])
print("Content recs:", list(content_recs)[:5])
print("Collab recs:", collab_recs[:5])
print("Hybrid recs:", list(hybrid_recs)[:5])

# ----------------------------
# EVALUATION
# ----------------------------
evaluate_models(
    list(content_recs),
    collab_recs,
    list(hybrid_recs),
    ground_truth
)

# =========================
# MULTI-USER EVALUATION
# =========================

print("\nRunning Multi-User Evaluation...")

user_ids = ratings['userId'].unique()[:10]  # take 10 users

content_precisions = []
collab_precisions = []
hybrid_precisions = []

k = 10

for uid in user_ids:

    user_test_data = test[test['userId'] == uid]
    ground_truth = user_test_data[user_test_data['rating'] >= 4]['title'].tolist()

    if len(ground_truth) == 0:
        continue

    liked_movie = ground_truth[0]

    # Content
    content_recs = content_model.recommend(liked_movie, movies)

    # Collaborative
    collab_recs = get_collab_recommendations(collab_model, uid, movies, train)

    # Hybrid
    movie_idx = movies[movies['title'] == liked_movie].index[0]

    hybrid_recs = hybrid_model.recommend(
        uid,
        movies,
        content_model.similarity_matrix[movie_idx]
    )

    # Metrics
    content_precisions.append(
        precision_at_k(list(content_recs), ground_truth, k)
    )

    collab_precisions.append(
        precision_at_k(collab_recs, ground_truth, k)
    )

    hybrid_precisions.append(
        precision_at_k(list(hybrid_recs), ground_truth, k)
    )

# -------------------------
# AVERAGE RESULTS
# -------------------------

print("\nAverage Precision@10 (across users):")

print("Content:", sum(content_precisions)/len(content_precisions))
print("Collaborative:", sum(collab_precisions)/len(collab_precisions))
print("Hybrid:", sum(hybrid_precisions)/len(hybrid_precisions))

# ----------------------------
# PLOTTING
# ----------------------------
k = 10

models = ["Content", "Collaborative", "Hybrid"]

precision = [
    precision_at_k(list(content_recs), ground_truth, k),
    precision_at_k(collab_recs, ground_truth, k),
    precision_at_k(list(hybrid_recs), ground_truth, k)
]

recall = [
    recall_at_k(list(content_recs), ground_truth, k),
    recall_at_k(collab_recs, ground_truth, k),
    recall_at_k(list(hybrid_recs), ground_truth, k)
]

plt.figure()
plt.bar(models, precision)
plt.title("Precision@10 Comparison")
plt.show()

plt.figure()
plt.bar(models, recall)
plt.title("Recall@10 Comparison")
plt.show()


# =========================
# ALPHA TUNING EXPERIMENT
# =========================

print("\nRunning Alpha Tuning Experiment...")

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_scores = []

user_ids = ratings['userId'].unique()[:10]
k = 10

for alpha in alphas:
    hybrid_model = HybridModel(content_model, collab_model, alpha=alpha)

    precisions = []

    for uid in user_ids:
        user_test_data = test[test['userId'] == uid]
        ground_truth = user_test_data[user_test_data['rating'] >= 4]['title'].tolist()

        if len(ground_truth) == 0:
            continue

        liked_movie = ground_truth[0]

        movie_idx = movies[movies['title'] == liked_movie].index[0]

        hybrid_recs = hybrid_model.recommend(
            uid,
            movies,
            content_model.similarity_matrix[movie_idx]
        )

        precisions.append(
            precision_at_k(list(hybrid_recs), ground_truth, k)
        )

    avg_precision = sum(precisions) / len(precisions)
    alpha_scores.append(avg_precision)

# Plot
plt.figure()
plt.plot(alphas, alpha_scores, marker='o')
plt.xlabel("Alpha (Content Weight)")
plt.ylabel("Precision@10")
plt.title("Effect of Alpha on Hybrid Model")
plt.show()
