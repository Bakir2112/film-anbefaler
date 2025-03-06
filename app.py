import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 📂 Indlæs filmen fra CSV
movies_df = pd.read_csv("movies.csv")

# 🎭 Konverter genrer til tal
label_encoder = LabelEncoder()
movies_df["genre"] = label_encoder.fit_transform(movies_df["genre"])

# 🎯 Standardiser numeriske features
scaler = StandardScaler()
X = scaler.fit_transform(movies_df[["genre", "rating"]])  # Kun numeriske værdier

# 🔍 Træn KNN-modellen
knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(X)

# 🎬 Film-anbefaling funktion
def recommend_movies(movie_title):
    if movie_title not in movies_df["title"].values:
        return "Film ikke fundet."

    movie_idx = movies_df[movies_df["title"] == movie_title].index[0]
    distances, indices = knn.kneighbors([X[movie_idx]])
    recommendations = movies_df.iloc[indices[0][1:]]
    return recommendations[["title", "rating"]]

# 🎥 Streamlit UI
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Vælg en film:", movies_df["title"])

if st.button("Få anbefalinger"):
    recommended_movies = recommend_movies(selected_movie)
    st.write("**Anbefalede film:**")
    st.write(recommended_movies)
