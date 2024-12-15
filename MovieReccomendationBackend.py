import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_top_movies(connection: sqlite3.Connection) -> pd.DataFrame:
    try:
        query = """
        SELECT 
            movies.title, 
            AVG(ratings.rating) AS avg_rating, 
            COUNT(ratings.rating) AS num_ratings
        FROM 
            movies
        JOIN 
            ratings ON movies.movieid = ratings.movieid
        GROUP BY 
            movies.movieid
        HAVING 
            COUNT(ratings.rating) > 50
        ORDER BY 
            avg_rating DESC
        LIMIT 10;
        """
        return pd.read_sql_query(query, connection)
    except Exception as e:
        print(f"Error fetching top movies: {e}")
        return pd.DataFrame()

def get_movie_data(movies_df:pd.DataFrame) -> pd.DataFrame:
    """
    Process the movie DataFrame to prepare it for recommendations.

    Args:
        movies_df (pd.DataFrame): Original DataFrame containing movie data.

    Returns:
        pd.DataFrame: Processed DataFrame with genre data split and normalized.
    """
    movies_df['genres_split'] = movies_df['genres'].str.split('|')
    movies_df['genres_str'] = movies_df['genres_split'].apply(lambda x: " ".join(x))
    return movies_df

def normalise_title(title:str) -> str:
    """
    Normalize a movie title by removing the year and converting to lowercase.

    Args:
        title (str): The original movie title.

    Returns:
        str: The normalized title.
    """
    return title.split('(')[0].strip().lower()

def find_movie_by_title(user_input: str, movies_df: pd.DataFrame) -> int:
    """
    Find the movie index based on a user input that is case-insensitive 
    and ignores the year in the title.

    Args:
        user_input (str): The user's movie title input.
        movies_df (pd.DataFrame): The DataFrame containing movie titles.

    Returns:
        int: The index of the matched movie.
    """
    # Normalize user input
    normalized_input = user_input.strip().lower()

    # Normalize all movie titles in the DataFrame
    movies_df['normalized_title'] = movies_df['title'].apply(normalise_title)

    # Search for the movie
    matches = movies_df[movies_df['normalized_title'] == normalized_input]
    if matches.empty:
        raise ValueError(f"No movie found matching '{user_input}'.")
    
    return matches.index[0]


def get_content_recommendations(film_title: str, movies_df: pd.DataFrame) -> list:
    """
    Generate content-based movie recommendations using cosine similarity.

    Args:
        film_title (str): The title of the movie to base recommendations on.
        movies_df (pd.DataFrame): DataFrame containing movie details.

    Returns:
        list: A sorted list of similar movies with their similarity scores.
    """
    try:
        # Find the movie index using flexible matching
        movie_index = find_movie_by_title(film_title, movies_df)

        # Calculate similarity
        vectorizer = CountVectorizer()
        genre_matrix = vectorizer.fit_transform(movies_df['genres_str'])
        similarity = cosine_similarity(genre_matrix)

        # Sort by similarity
        similar_movies = list(enumerate(similarity[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        return similar_movies[1:6]  # Skip the first movie (itself)
    except Exception as e:
        print(f"Error in content-based recommendations: {e}")
        return []

def get_user_recommendations(user_id: str, ratings_df: pd.DataFrame):
    """
    Generate user-based movie recommendations using collaborative filtering.

    calculates the similarity between users based on their movie
    ratings and finds users most similar to the given `user_id`. It returns a 
    list of the top similar users along with their similarity scores.

    Args:
        user_id (int): The ID of the target user for whom recommendations are to be generated.
        ratings_df (pd.DataFrame): A DataFrame containing user ratings with columns:
            - 'user_id': The unique ID of each user.
            - 'movie_id': The unique ID of each movie.
            - 'rating': The rating given by the user to the movie.

    Returns:
        list: A sorted list of tuples, where each tuple contains:
            - The index of a similar user.
            - The similarity score (float) between the target user and the similar user.
        If the `user_id` is not found in the dataset, an empty list is returned.

    Raises:
        ValueError: If the provided `user_id` is not found in the `ratings_df`.

    Example:
        >>> ratings_data = pd.DataFrame({
                'user_id': [1, 2, 1, 2, 3],
                'movie_id': [101, 101, 102, 103, 102],
                'rating': [4.0, 5.0, 3.0, 2.0, 4.5]
            })
        >>> get_user_recommendations(1, ratings_data)
        [(2, 0.89), (3, 0.75)]
    """

    ratings_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    ratings_matrix = ratings_matrix.fillna(0)
    user_similarity = cosine_similarity(ratings_matrix)
    if user_id in ratings_matrix.index:
        similar_users = user_similarity[user_id]
        top_users = sorted(list(enumerate(similar_users)), key=lambda x: x[1], reverse=True)
        print(f"Top users similar to user {user_id}: {top_users[:5]}")
    else:
        print(f"User ID {user_id} not found in the dataset.")
        
    return sorted(list(enumerate(similar_users)), key=lambda x: x[1], reverse=True)

def main(film_title:str, user_id:int):
    try:
        with sqlite3.connect('movies.db') as connection:
            movies_df = pd.read_csv('movies.csv')
            ratings_df = pd.read_csv('ratings.csv')

            # Process movie data
            movies_df = get_movie_data(movies_df)

            # Fetch and display top movies
            top_movies = fetch_top_movies(connection)
            print("Top Movies:")
            print(top_movies)

            # Example: Content-based recommendations
            if film_title != '':
                film_title = normalise_title(film_title)
                similar_movies = get_content_recommendations(film_title, movies_df)
                print(f"\nMovies similar to '{film_title}':")
                for idx, score in similar_movies:
                    print(f"{movies_df.iloc[idx].title} (Similarity: {score:.2f})")

            # Example: User-based recommendations
            if user_id != '':
                similar_users = get_user_recommendations(user_id, ratings_df)
                print(f"\nTop users similar to user {user_id}:")
                print(similar_users)
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main(
        film_title='heat',
        user_id=''
    
    )