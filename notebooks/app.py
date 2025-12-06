import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime

# Simple user database (in real app, use proper database)
USER_FILE = "users.json"

def load_users():
    try:
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f)

def load_user_ratings():
    try:
        return pd.read_csv("user_ratings.csv")
    except:
        return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

def save_user_rating(user_id, movie_id, rating):
    user_ratings = load_user_ratings()
    new_rating = pd.DataFrame({
        'userId': [user_id],
        'movieId': [movie_id],
        'rating': [rating],
        'timestamp': [int(datetime.now().timestamp())]
    })
    user_ratings = pd.concat([user_ratings, new_rating], ignore_index=True)
    user_ratings.to_csv("user_ratings.csv", index=False)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

# LOGIN SCREEN
if not st.session_state['logged_in']:
    st.title("Movie Recommendation System")
    st.markdown("### Please login or sign up to continue")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", type="primary"):
            users = load_users()
            if login_username in users and users[login_username]['password'] == login_password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = login_username
                st.session_state['user_id'] = users[login_username]['user_id']
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Sign Up")
        signup_username = st.text_input("Choose Username", key="signup_user")
        signup_password = st.text_input("Choose Password", type="password", key="signup_pass")
        signup_password2 = st.text_input("Confirm Password", type="password", key="signup_pass2")
        
        if st.button("Sign Up", type="primary"):
            users = load_users()
            if signup_username in users:
                st.error("Username already exists")
            elif signup_password != signup_password2:
                st.error("Passwords don't match")
            elif len(signup_username) < 3 or len(signup_password) < 3:
                st.error("Username and password must be at least 3 characters")
            else:
                # Create new user
                user_id = max([u['user_id'] for u in users.values()], default=0) + 1
                users[signup_username] = {
                    'password': signup_password,
                    'user_id': user_id
                }
                save_users(users)
                st.success("Account created! Please login.")
    
    st.stop()  # Stop execution until logged in

# MAIN APP (After Login)
# Loading Data
@st.cache_data
def load_data():
    movies = pd.read_csv("../data/movies.csv")
    ratings = pd.read_csv("../data/ratings.csv")
    return movies, ratings

movies, ratings = load_data()

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title.lower()

@st.cache_data
def create_tfidf(_movies):
    movies_copy = _movies.copy()
    movies_copy['clean_title'] = movies_copy['title'].apply(clean_title)
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(movies_copy['clean_title'])
    return vectorizer, tfidf, movies_copy

vectorizer, tfidf, movies = create_tfidf(movies)

# Header with logout
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Movie Recommendation System")
    st.markdown(f"### Welcome, {st.session_state['username']}!")
with col2:
    if st.button("Logout", type="secondary"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['user_id'] = None
        st.rerun()

movie_title = st.text_input("Enter a movie title:", placeholder="e.g, Toy Story, The Avengers")

# Search Functionality
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies.iloc[indices].copy()
    
    movie_ratings_count = ratings.groupby('movieId').size().reset_index(name='rating_count')
    results = results.merge(movie_ratings_count, on='movieId', how='left')
    results['rating_count'] = results['rating_count'].fillna(0)
    results['similarity'] = similarity[indices]
    
    results['log_ratings'] = np.log1p(results['rating_count'])
    max_log_ratings = results['log_ratings'].max()
    if max_log_ratings > 0:
        results['normalized_ratings'] = results['log_ratings'] / max_log_ratings
    else:
        results['normalized_ratings'] = 0
    
    results['combined_score'] = 0.4 * results['similarity'] + 0.6 * results['normalized_ratings']
    results = results.sort_values('combined_score', ascending=False)
    
    return results.head(5)

# Collaborative Filtering
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings['movieId'] == movie_id) & (ratings['rating'] > 4)]["userId"].unique()
    
    if len(similar_users) == 0:
        return pd.DataFrame({"message": ["Not enough data for recommendations"]})
    
    similar_user_recs = ratings[(ratings['userId'].isin(similar_users)) & (ratings['rating'] > 4)]
    similar_user_recs = similar_user_recs['movieId'].value_counts() / len(similar_users)
    popular_movies = similar_user_recs[similar_user_recs > 0.05]
    
    if len(popular_movies) == 0:
        return pd.DataFrame({"message": ["Not enough data for recommendations"]})
    
    all_users = ratings[(ratings["movieId"].isin(popular_movies.index)) & (ratings['rating'] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1, join='inner')
    
    if len(rec_percentages) == 0:
        return pd.DataFrame({"message": ["Not enough data for recommendations"]})
    
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    result = rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    
    avg_ratings = ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating', 'num_ratings']
    result = result.merge(movies[['movieId', 'title']], on='title', how='left')
    result = result.merge(avg_ratings, on='movieId', how='left')
    
    return result[["movieId", "title", "genres", "score", "avg_rating", "num_ratings"]]

# Session state for recommendations
if 'search_done' not in st.session_state:
    st.session_state['search_done'] = False
if 'movie_id' not in st.session_state:
    st.session_state['movie_id'] = None
if 'movie_name' not in st.session_state:
    st.session_state['movie_name'] = None

if st.button("Search & Get Recommendations", type="primary"):
    if len(movie_title) > 5:
        with st.spinner("Searching for movies..."):
            results = search(movie_title)
            
            if len(results) > 0:
                st.session_state.movie_id = results.iloc[0]['movieId']
                st.session_state.movie_name = results.iloc[0]['title']
                st.session_state.search_done = True
                st.success(f"Found: **{st.session_state.movie_name}**")
            else:
                st.error("No movies found. Try a different search.")
                st.session_state.search_done = False
    else:
        st.warning("Please enter at least 6 characters")

# Display recommendations
if st.session_state.search_done:
    st.markdown("---")
    
    # Rate the searched movie
    st.subheader(f"Rate: {st.session_state.movie_name}")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_rating = st.slider("Your rating:", 0.5, 5.0, 3.0, 0.5, key="search_rating")
    with col2:
        if st.button("Submit Rating", type="primary"):
            save_user_rating(st.session_state['user_id'], st.session_state.movie_id, user_rating)
            st.success("Rating saved!")
    
    st.markdown("---")
    st.subheader(f"Recommendations for: {st.session_state.movie_name}")
    
       
    with st.spinner("Analyzing user preferences..."):
        collab_recs = find_similar_movies(st.session_state.movie_id)
        
        if 'message' not in collab_recs.columns:
            for idx, row in collab_recs.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Genres: {row['genres']}")
                        st.caption(f"Match Score: {row['score']:.3f} | Rating: {row['avg_rating']:.2f} ({int(row['num_ratings'])} reviews)")
                    with col2:
                        rating_key = f"rate_collab_{row['movieId']}"
                        rating = st.slider("Rate:", 0.5, 5.0, 3.0, 0.5, key=rating_key, label_visibility="collapsed")
                        if st.button("Rate", key=f"btn_collab_{row['movieId']}", type="secondary"):
                            save_user_rating(st.session_state['user_id'], row['movieId'], rating)
                            st.success("✓")
                    st.markdown("---")
        else:
            st.warning("Not enough collaborative filtering data for this movie. Try searching for a more popular movie!")

st.markdown("---")
st.markdown("**Tip:** Rate movies to improve future recommendations!")