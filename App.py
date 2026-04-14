import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommender System")
st.write("Find similar movies, combine multiple favorites, or browse top movies by genre.")

# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_merged.csv")

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies["genres_clean"] = movies["genres_clean"].fillna("").astype(str)

    return movies

@st.cache_resource
def build_features(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    return tfidf_matrix, indices

movies = load_data()
tfidf_matrix, indices = build_features(movies)

# -----------------------------------
# Helper functions
# -----------------------------------
def rerank(df, alpha=0.8):
    df = df.copy()

    sim_min, sim_max = df["similarity_score"].min(), df["similarity_score"].max()
    if sim_max > sim_min:
        df["sim_norm"] = (df["similarity_score"] - sim_min) / (sim_max - sim_min)
    else:
        df["sim_norm"] = 0

    r_min, r_max = df["avg_rating"].min(), df["avg_rating"].max()
    if r_max > r_min:
        df["rating_norm"] = (df["avg_rating"] - r_min) / (r_max - r_min)
    else:
        df["rating_norm"] = 0

    df["final_score"] = alpha * df["sim_norm"] + (1 - alpha) * df["rating_norm"]
    return df.sort_values(["final_score", "num_ratings"], ascending=[False, False])


def get_recommendations(title, top_n=10, min_shared_genres=1, alpha=0.8, min_ratings=20):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    target_genres = set(movies.loc[idx, "genres_clean"].split())

    mask = movies["genres_clean"].apply(
        lambda g: len(target_genres & set(g.split())) >= min_shared_genres
    )
    mask.iloc[idx] = False
    candidate_idx = movies[mask].index

    sim_scores_array = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix[candidate_idx]
    ).flatten()

    recommended = movies.loc[candidate_idx].copy()
    recommended["similarity_score"] = sim_scores_array
    recommended = recommended[recommended["num_ratings"] >= min_ratings]

    recommended = rerank(recommended, alpha=alpha)

    return recommended[
        ["title", "genres", "similarity_score", "avg_rating", "num_ratings", "final_score"]
    ].head(top_n)


def get_recommendations_from_list(titles, top_n=10, min_shared_genres=1, alpha=0.8, min_ratings=20):
    valid_titles = [t for t in titles if t in indices]

    if not valid_titles:
        return pd.DataFrame()

    input_idx = [indices[t] for t in valid_titles]

    user_profile = tfidf_matrix[input_idx].mean(axis=0)
    user_profile = np.asarray(user_profile).reshape(1, -1)

    sim_scores_array = cosine_similarity(user_profile, tfidf_matrix).flatten()

    recommended = movies.copy()
    recommended["similarity_score"] = sim_scores_array
    recommended = recommended[~recommended.index.isin(input_idx)].copy()

    target_genres = set()
    for idx in input_idx:
        target_genres |= set(movies.loc[idx, "genres_clean"].split())

    mask = recommended["genres_clean"].apply(
        lambda g: len(target_genres & set(g.split())) >= min_shared_genres
    )
    recommended = recommended[mask]
    recommended = recommended[recommended["num_ratings"] >= min_ratings]

    recommended = rerank(recommended, alpha=alpha)

    return recommended[
        ["title", "genres", "similarity_score", "avg_rating", "num_ratings", "final_score"]
    ].head(top_n)


def get_top_movies_by_genres(selected_genres, top_n=10, min_votes=20):
    if not selected_genres:
        return pd.DataFrame()

    selected_genres = set(g.lower() for g in selected_genres)

    filtered = movies[movies["genres_clean"].apply(
        lambda g: selected_genres.issubset(set(g.split()))
    )].copy()

    filtered = filtered[filtered["num_ratings"] >= min_votes]

    filtered = filtered.sort_values(
        by=["avg_rating", "num_ratings"],
        ascending=[False, False]
    )

    return filtered[["title", "genres", "avg_rating", "num_ratings"]].head(top_n)

# -----------------------------------
# Sidebar controls
# -----------------------------------
st.sidebar.header("Options")

mode = st.sidebar.radio(
    "Choose recommendation mode",
    [
        "Similar to one movie",
        "Similar to multiple movies",
        "Top movies by genre"
    ]
)

top_n = st.sidebar.slider("Number of results", min_value=5, max_value=20, value=10)
min_ratings = st.sidebar.slider("Minimum number of ratings", min_value=0, max_value=200, value=20)

# Only show these when relevant
if mode in ["Similar to one movie", "Similar to multiple movies"]:
    min_shared_genres = st.sidebar.slider("Minimum shared genres", min_value=1, max_value=3, value=1)
    alpha = st.sidebar.slider("Similarity vs rating balance", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

# -----------------------------------
# Main UI
# -----------------------------------
if mode == "Similar to one movie":
    st.subheader("Recommend movies similar to one title")

    selected_title = st.selectbox(
        "Choose a movie",
        sorted(movies["title"].dropna().unique())
    )

    if st.button("Get recommendations", key="single_movie_btn"):
        results = get_recommendations(
            title=selected_title,
            top_n=top_n,
            min_shared_genres=min_shared_genres,
            alpha=alpha,
            min_ratings=min_ratings
        )

        if results.empty:
            st.warning("No recommendations found.")
        else:
            st.success(f"Showing recommendations similar to: {selected_title}")
            st.dataframe(results, use_container_width=True)


elif mode == "Similar to multiple movies":
    st.subheader("Recommend movies based on multiple favorites")

    selected_titles = st.multiselect(
        "Choose two or more movies",
        sorted(movies["title"].dropna().unique())
    )

    if st.button("Get combined recommendations", key="multi_movie_btn"):
        if len(selected_titles) < 2:
            st.warning("Please select at least two movies.")
        else:
            results = get_recommendations_from_list(
                titles=selected_titles,
                top_n=top_n,
                min_shared_genres=min_shared_genres,
                alpha=alpha,
                min_ratings=min_ratings
            )

            if results.empty:
                st.warning("No recommendations found.")
            else:
                st.success("Showing recommendations based on your selected movies.")
                st.write("Selected movies:", ", ".join(selected_titles))
                st.dataframe(results, use_container_width=True)


elif mode == "Top movies by genre":
    st.subheader("Browse top movies by genre")

    all_genres = sorted({
        genre
        for genre_list in movies["genres_clean"].dropna().str.split()
        for genre in genre_list
    })

    selected_genres = st.multiselect(
        "Choose one or more genres",
        all_genres
    )

    if st.button("Show top movies", key="genre_btn"):
        if not selected_genres:
            st.warning("Please select at least one genre.")
        else:
            results = get_top_movies_by_genres(
                selected_genres=selected_genres,
                top_n=top_n,
                min_votes=min_ratings
            )

            if results.empty:
                st.warning("No movies found for that genre combination.")
            else:
                st.success(f"Showing top movies for: {', '.join(selected_genres)}")
                st.dataframe(results, use_container_width=True)

# -----------------------------------
# Footer note
# -----------------------------------
st.markdown("---")
st.caption(
    "This app uses a content-based recommender built with TF-IDF, cosine similarity, "
    "genre filtering, and reranking with average rating."
)
