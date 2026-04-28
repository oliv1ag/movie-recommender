# 🎬 Hybrid Movie Recommendation System

A machine learning project that builds a **hybrid movie recommendation system** by combining **content-based filtering (NLP)** and **collaborative filtering (SVD)**. The system generates personalized movie recommendations and includes evaluation metrics and an interactive UI.

---

## 🚀 Features

* 🔍 **Content-Based Filtering**

  * Uses TF-IDF on movie genres (as proxy for text features)
  * Computes cosine similarity between movies

* 🤝 **Collaborative Filtering**

  * Uses **SVD (matrix factorization)** via Surprise library
  * Learns user preferences from rating patterns

* 🔥 **Hybrid Recommendation Model**

  * Combines both approaches:

    ```
    score = α × content_score + (1 - α) × collaborative_score
    ```
  * Improves recommendation quality over individual models

* 📊 **Evaluation Metrics**

  * RMSE (for rating prediction)
  * Precision@K
  * Recall@K

* 💻 **Interactive UI (Streamlit)**

  * Select user and movie
  * Adjust hybrid weight (alpha)
  * View recommendations from all models

---

## 📂 Project Structure

```
ML_Project/
│
├── data_loader.py
├── content_model.py
├── collaborative_model.py
├── hybrid_model.py
├── evaluation.py
├── main.py
├── app.py
├── ratings.csv
├── movies.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/oliv1ag/movie-recommender.git
cd movie-recommender
```

2. Create virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Run ML pipeline:

```bash
python main.py
```

### Run UI:

```bash
streamlit run app.py
```

---

## 📊 Sample Results

| Model         | Precision@10 | Recall@10 |
| ------------- | ------------ | --------- |
| Content-Based | 0.10         | 0.03      |
| Collaborative | 0.00         | 0.00      |
| Hybrid        | **0.20**     | **0.06**  |

👉 Hybrid model improves recommendation performance.

---

## 🧠 Approach

### 1. Content-Based Filtering

* Converts genres into text features
* Applies TF-IDF vectorization
* Uses cosine similarity to find similar movies

### 2. Collaborative Filtering

* Constructs user-item matrix
* Applies SVD to learn latent features
* Predicts ratings for unseen movies

### 3. Hybrid Model

* Combines both scores using weighted sum
* Balances similarity and user preference

---

## ❄️ Cold Start Handling

* Content-based filtering enables recommendations for:

  * New users (no history)
  * New movies (no ratings)

---

## ⚠️ Limitations

* MovieLens dataset does not include plot descriptions
* Genres used as proxy for textual features
* Evaluation performed on limited user samples

---

## 🔮 Future Improvements

* Use **BERT embeddings** for richer NLP features
* Add **movie metadata (cast, crew, year)**
* Evaluate across multiple users
* Deploy as full web application

---

## 🛠️ Tech Stack

* Python
* pandas, numpy
* scikit-learn
* scikit-surprise
* matplotlib
* Streamlit

---



## ⭐ Acknowledgements

* MovieLens Dataset
* Surprise Library
* scikit-learn

---
