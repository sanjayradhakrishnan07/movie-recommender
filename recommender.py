import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.ratings = None
        self.user_similarity = None
        self.movie_similarity = None

    def load_data(self, ratings_file):
        """
        Load movie ratings data from a CSV file.
        The file should contain user IDs, movie IDs, and ratings.
        """
        self.ratings = pd.read_csv(ratings_file)
        self.ratings = self.ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    def calculate_user_similarity(self):
        """
        Calculate user similarity matrix based on cosine similarity.
        """
        self.user_similarity = cosine_similarity(self.ratings)

    def generate_recommendations(self, user_id, num_recommendations=5):
        """
        Generate movie recommendations for a given user.
        Returns a list of recommended movie IDs.
        """
        if self.user_similarity is None:
            self.calculate_user_similarity()

        user_index = self.ratings.index.get_loc(user_id)
        similar_users = list(enumerate(self.user_similarity[user_index]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]

        recommended_movie_ids = set()
        for index, _ in similar_users:
            movie_ids = self.ratings.iloc[index][self.ratings.iloc[index] > 0].index
            recommended_movie_ids.update(movie_ids)
        # Exclude movies already rated by the user
        rated_movies = self.ratings.loc[user_id][self.ratings.loc[user_id] > 0].index
        recommended_movie_ids.difference_update(rated_movies)

        # Return top N recommended movie IDs
        return list(recommended_movie_ids)[:num_recommendations]