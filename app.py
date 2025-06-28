import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
df = pd.read_csv("movie_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Initialize session state variables
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "recommend_for" not in st.session_state:
    st.session_state.recommend_for = {}
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Page config
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Sidebar navigation with selectbox instead of radio buttons
page = st.sidebar.selectbox("Choose a Page", ["🏠 Home", "❤️ Saved Movies"])

# TF-IDF vectorization for recommendations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, num=5):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def display_movies(dataframe, section_id=""):
    for i, row in dataframe.iterrows():
        unique_suffix = f"{section_id}_{i}"
        cols = st.columns([1, 2, 1])
        with cols[0]:
            st.image(row.get("poster_url", ""), width=120)
        with cols[1]:
            st.subheader(row.get("title", ""))
            st.write(f"Genre: {row.get('genre', '')}")
            st.write(f"⭐ Rating: {row.get('rating', 'N/A')} | 🔥 Popularity: {row.get('popularity', 'N/A')}")
        with cols[2]:
            if st.button("❤️ Save", key=f"save_{unique_suffix}"):
                if row['title'] not in st.session_state.favorites:
                    st.session_state.favorites.append(row['title'])
                    st.success(f"✅ '{row['title']}' has been saved to your favorites!")
            if st.button("🎯 Recommend", key=f"rec_{unique_suffix}"):
                recs = get_recommendations(row['title'], num=5)
                st.session_state.recommend_for[row['title']] = recs

        if row['title'] in st.session_state.recommend_for:
            st.markdown("---")
            with st.container():
                st.markdown(f"<h4 style='margin-top:0;'>🎯 Movies similar to '{row['title']}'</h4>", unsafe_allow_html=True)
                recs = st.session_state.recommend_for[row['title']]
                scroll_html = """<div style='display: flex; overflow-x: auto; gap: 2rem;'>"""
                for j, rec in recs.iterrows():
                    card_html = f"""
                    <div style='min-width: 180px; background: #111; padding: 10px; border-radius: 10px; color: white;'>
                        <img src='{rec.get("poster_url", "").strip()}' style='width: 100%; border-radius: 8px;'>
                        <p style='margin-top: 10px; font-weight: bold;'>{rec.get("title", "")}</p>
                        <small>Genre: {rec.get("genre", "")}<br>⭐ {rec.get("rating", "N/A")} | 🔥 {rec.get("popularity", "N/A")}</small>
                    </div>"""
                    scroll_html += card_html
                scroll_html += "</div>"
                st.markdown(scroll_html, unsafe_allow_html=True)

# HOME PAGE
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: white;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.write("### Get your favorite English movie recommendations below")

    # Search bar
    search_query = st.text_input("🔍 Search for a movie title")
    if search_query:
        search_query_lower = search_query.strip().lower()
        matches = df[df['title'].str.lower().str.contains(search_query_lower)]
        if not matches.empty:
            st.subheader(f"Search results for '{search_query}':")
            display_movies(matches, section_id="search")
        else:
            st.warning(f"No results found for '{search_query}'.")
            if st.button("💡 Suggest this movie for future addition"):
                if search_query not in st.session_state.suggestions:
                    st.session_state.suggestions.append(search_query)
                    st.success("Thanks! We've noted your suggestion.")
                else:
                    st.info("You already suggested this movie.")

    # Filters
    sort_option = st.sidebar.selectbox("Sort movies by", ["None", "Popularity", "Rating"])
    genre_filter = st.sidebar.multiselect("Filter by Genre", options=sorted(df['genre'].unique()))

    # Filter by Genre
    if genre_filter:
        filtered_df = df[df['genre'].isin(genre_filter)]
    else:
        filtered_df = df

    # Sort movies
    if sort_option == "Popularity":
        filtered_df = filtered_df.sort_values(by="popularity", ascending=False)
    elif sort_option == "Rating":
        filtered_df = filtered_df.sort_values(by="rating", ascending=False)

    st.subheader("Browse Movies")
    display_movies(filtered_df.head(12), section_id="main")

# SAVED MOVIES PAGE
elif page == "❤️ Saved Movies":
    st.markdown("<h2>❤️ My Saved Favorite Movies</h2>", unsafe_allow_html=True)
    if st.session_state.favorites:
        fav_df = df[df['title'].isin(st.session_state.favorites)]
        display_movies(fav_df, section_id="favorites")

        csv_fav = fav_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Favorite Movies as CSV",
            data=csv_fav,
            file_name='favorite_movies.csv',
            mime='text/csv'
        )
    else:
        st.info("You have not saved any movies yet.")

