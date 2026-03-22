import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    """
    A simple collaborative filtering movie recommendation system.
    Uses user similarity to recommend movies based on ratings.
    """
    
    def __init__(self, data_path='data/sample_ratings.csv'):
        """Initialize the recommender with data from a CSV file."""
        self.data = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load movie ratings data from CSV."""
        self.data = pd.read_csv(data_path)
        self.data.set_index('user_id', inplace=True)
        self.user_item_matrix = self.data.copy()
        print("✅ Data loaded successfully!")
        print(f"\nUser-Item Matrix:\n{self.user_item_matrix}\n")
    
    def calculate_user_similarity(self):
        """Calculate similarity between all users using cosine similarity."""
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        self.similarity_matrix = pd.DataFrame(
            self.similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        return self.similarity_matrix
    
    def get_recommendations(self, user_id, n_recommendations=3):
        """
        Generate top N movie recommendations for a user.
        
        Args:
            user_id: The ID of the user
            n_recommendations: Number of movies to recommend
        
        Returns:
            List of (movie, predicted_rating) tuples
        """
        if self.similarity_matrix is None:
            self.calculate_user_similarity()
        
        # Get similarity scores for this user with all other users
        user_similarities = self.similarity_matrix[user_id].drop(user_id)
        
        # Get movies the user has NOT rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        # If user hasn't rated any movies with 0, get all movies they haven't rated yet
        if not unrated_movies:
            unrated_movies = user_ratings.index.tolist()
        
        recommendations = {}
        
        # Calculate predicted ratings for unrated movies
        for movie in unrated_movies:
            # Get ratings from similar users for this movie
            similar_users_ratings = []
            similarities = []
            
            for other_user in self.user_item_matrix.index:
                if other_user != user_id:
                    rating = self.user_item_matrix.loc[other_user, movie]
                    similarity = user_similarities[other_user]
                    
                    if rating > 0:  # Only consider rated movies
                        similar_users_ratings.append(rating)
                        similarities.append(similarity)
            
            # Calculate weighted average
            if similarities:
                predicted_rating = np.average(similar_users_ratings, weights=similarities)
                recommendations[movie] = predicted_rating
        
        # Sort by predicted rating and return top N
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return sorted_recommendations

def main():
    """Main function to demonstrate the recommender."""
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDER SYSTEM 🎬")
    print("=" * 60)
    
    # Initialize recommender
    recommender = MovieRecommender('data/sample_ratings.csv')
    
    # Calculate similarities
    print("📊 Calculating user similarities...\n")
    recommender.calculate_user_similarity()
    
    # Generate recommendations for each user
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    for user_id in recommender.user_item_matrix.index:
        recommendations = recommender.get_recommendations(user_id, n_recommendations=3)
        
        print(f"\nTop 3 Movie Recommendations for User {user_id}:\n")
        print("-" * 60)
        
        if recommendations:
            for idx, (movie, rating) in enumerate(recommendations, 1):
                print(f"{idx}. {movie:30} - Score: {rating:.2f}/5 ⭐")
        else:
            print("No recommendations available.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()