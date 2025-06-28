# 🎬 Movie Recommendation System

This is a content-based movie recommendation system built using **Streamlit** and **scikit-learn**. It allows users to:

- Search for movies
- Filter by genre, rating, and popularity
- View recommendations for each movie
- Save favorite movies
- Download favorite movies as a CSV
- Suggest missing movies for future updates

---

## 📊 Dataset

The dataset (`movie_dataset.csv`) includes:

- `title`: Movie title  
- `genre`: Genre of the movie  
- `poster_url`: Link to the movie poster image  
- `rating`: IMDB-style rating  
- `popularity`: A numerical score representing popularity  

---

## 💻 How to Run

### Option 1: Try it online 
👉 Click here to try the app(https://movie-recommendation-system-rffar6csmubw7fjkeygmjt.streamlit.app/)

### Option 2: Run via Git and Streamlit (locally)
git clone https://github.com/kavyacodes08/movie-recommendation-system.git
```bash
Commands
cd movie-recommendation-system
pip install -r requirements.txt
streamlit run app.py
