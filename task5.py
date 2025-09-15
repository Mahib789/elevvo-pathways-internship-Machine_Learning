# task5.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

print("Ratings Data:")
print(ratings.head())
print("\nMovies Data:")
print(movies.head())

# Create user-item rating matrix
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print("\nUser-Item Matrix:")
print(user_item_matrix.head())

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("\nUser Similarity Matrix:")
print(user_similarity_df.head())

# Function to recommend movies for a user
def recommend_movies(user_id, num_recommendations=3):
    # Get similarity scores for this user
    sim_scores = user_similarity_df[user_id].drop(user_id)
    most_similar_user = sim_scores.idxmax()
    
    # Movies rated by the most similar user
    similar_user_ratings = ratings[ratings["userId"] == most_similar_user]
    target_user_ratings = ratings[ratings["userId"] == user_id]
    
    # Recommend movies the similar user liked but target user hasn’t seen
    recommendations = similar_user_ratings[~similar_user_ratings["movieId"].isin(target_user_ratings["movieId"])]
    top_recommendations = recommendations.sort_values("rating", ascending=False).head(num_recommendations)
    
    # Join with movies to get titles
    return top_recommendations.merge(movies, on="movieId")[["movieId", "title", "rating"]]

# Example: Recommend for user 1
print("\nMovie Recommendations for User 1:")
print(recommend_movies(1, num_recommendations=3))
