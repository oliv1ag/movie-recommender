import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

class CollaborativeModel:
    def __init__(self):
        self.model = SVD()
        self.trainset = None
        self.testset = None

    def fit(self, ratings_df):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], reader
        )

        self.trainset, self.testset = train_test_split(data, test_size=0.2)
        self.model.fit(self.trainset)

    def evaluate(self):
        predictions = self.model.test(self.testset)
        return rmse(predictions)

    def predict(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est