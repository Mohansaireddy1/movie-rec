import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Load data from the MovieLens dataset
ratings = pd.read_csv('path_to_ratings.csv')
movies = pd.read_csv('path_to_movies.csv')

# Merge movies and ratings
df = pd.merge(ratings, movies, on='movieId')

# Drop unnecessary columns
df = df.drop(['timestamp', 'genres'], axis=1)

# Initialize Reader and Dataset for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Build and train the SVD model
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train on the whole dataset
trainset = data.build_full_trainset()
svd.fit(trainset)

# Recommend movies for a user
def recommend_movies(user_id, n=10):
    movies_rated_by_user = df[df['userId'] == user_id]
    movies_not_rated = movies[~movies['movieId'].isin(movies_rated_by_user['movieId'])]
    
    predictions = []
    for movie_id in movies_not_rated['movieId']:
        pred = svd.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    recommended_movie_ids = [movie[0] for movie in recommendations]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    
    return recommended_movies

# Example usage
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Recommended movies for user {}: ".format(user_id))
print(recommended_movies[['movieId', 'title']])
