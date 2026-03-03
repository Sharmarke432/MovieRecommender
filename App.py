import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Page Config ---
st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="centered")
st.title("🍿 Content-Based Movie Recommender")
st.markdown("Find your next favorite movie based on genres and community ratings!")

# --- 2. Load Data (with Caching for speed) ---
@st.cache_data
def load_data():
    # Load the files we saved from Jupyter
    movies = pd.read_pickle("movies_df.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    
    # Create the reverse index mapping
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    return movies, tfidf_matrix, indices

movies, tfidf_matrix, indices = load_data()

# --- 3. Recommendation Function ---
def get_recommendations(title, top_n=5):
    idx = indices[title]
    
    # On-the-fly similarity
    sim_scores_array = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    recommended = movies.copy()
    recommended["similarity"] = sim_scores_array
    
    # Remove the searched movie
    recommended = recommended[recommended.index != idx]
    
    # Sort by similarity, then by average rating
    recommended = recommended.sort_values(
        by=["similarity", "avg_rating"], 
        ascending=[False, False]
    )
    
    return recommended.head(top_n)

# --- 4. Streamlit UI ---
# Dropdown menu for movie selection
selected_movie = st.selectbox(
    "Search for a movie you like:",
    options=movies["title"].values,
    index=None,
    placeholder="Start typing a movie name (e.g. Toy Story)..." 
)

# We only show the slider and button IF the user has actually selected a movie
if selected_movie:
    num_recs = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)
    
    if st.button("Recommend Movies"):
        with st.spinner("Finding the best matches..."):
            recs = get_recommendations(selected_movie, top_n=num_recs)
            
            st.success(f"Top {num_recs} recommendations for **{selected_movie}**:")
            
            # Display results nicely
            for _, row in recs.iterrows():
                with st.container():
                    st.subheader(row['title'])
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**Genres:** {row['genres']}")
                    col2.write(f"**Rating:** ⭐ {row['avg_rating']:.1f} ({row['num_ratings']} votes)")
                    col3.write(f"**Match:** {row['similarity']*100:.1f}%")
                    st.divider()