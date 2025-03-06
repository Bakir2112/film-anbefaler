import streamlit as st
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# TMDb API Setup
API_URL = "https://api.themoviedb.org/3"
API_KEY = "your_tmdb_api_key_here"

# Fetch movies from TMDb
def fetch_popular_movies():
    url = f"{API_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        movies = pd.DataFrame([{ 
            'title': m['title'], 
            'genre': m['genre_ids'][0] if m['genre_ids'] else 'Unknown',
            'rating': m['vote_average']
        } for m in data['results']])
        return movies
    else:
        return None

# Load movie data
movies_df = fetch_popular_movies()

# Train KNN model
if movies_df is not None:
    scaler = StandardScaler()
    X = scaler.fit_transform(movies_df[['genre', 'rating']])
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(X)

    def recommend_movies(movie_title):
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
        distances, indices = knn.kneighbors([X[movie_idx]])
        recommendations = movies_df.iloc[indices[0][1:]]
        return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")
if movies_df is not None:
    selected_movie = st.selectbox("Choose a movie:", movies_df['title'])
    if st.button("Get Recommendations"):
        recommended_movies = recommend_movies(selected_movie)
        st.write("**Recommended Movies:**")
        st.write(recommended_movies[['title', 'rating']])
else:
    st.write("Could not fetch movie data. Try again later.")
