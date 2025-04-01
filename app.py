import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load trained model and preprocessor
model = joblib.load("model.pkl")  
preprocessor = joblib.load("preprocessor.pkl")  

# Streamlit App UI
st.title("🎬 Movie Revenue Prediction App")
st.write("Enter movie details below to predict the box office revenue.")

# User Input Fields
budget = st.number_input("Budget ($)", min_value=1000000, step=1000000, value=50000000, max_value=1000000000)
running_time = st.number_input("Running Time (minutes)", min_value=60, max_value=400, step=5, value=150)
release_month = st.slider("Release Month", 1, 12, 6)
release_year = st.slider("Release Year", 1980, 2030, 2025)
director_popularity = st.number_input("Director Popularity ($ avg revenue)", min_value=0, value=50000000, max_value=1000000000)
lead_actor_popularity = st.number_input("Lead Actor Popularity ($ avg revenue)", min_value=0, value=50000000, max_value=1000000000)

holiday_season = st.selectbox("Holiday Season", ["Yes", "No"])
has_sequel = st.selectbox("Has Sequel", ["Yes", "No"])
franchise = st.selectbox("Franchise", ["None", "Avengers", "Harry Potter", "Star Wars", "Batman", "Spider-Man"])
genres = st.multiselect("Genres", 
                        ['Drama', 'Sport', 'Thriller', 'Fantasy', 'Animation', 'Crime', 'Sci-Fi', 'Action', 'Adventure', 'Romance', 'Documentary', 'Horror', 'Mystery', 'War', 'Family', 'Biography', 'Musical', 'Western', 'History', 'Comedy', 'Music']
                        )

# Convert categorical inputs to numerical values
holiday_season = 1 if holiday_season == "Yes" else 0
has_sequel = 1 if has_sequel == "Yes" else 0
franchise = 1 if franchise in ["Avengers", "Harry Potter", "Star Wars", "Batman", "Spider-Man"] else 0

# Prepare user input as a DataFrame
user_input = pd.DataFrame([{ 
    "Budget": budget,
    "Running Time": running_time,
    "Release Month": release_month,
    "Release Year": release_year,
    "Director Popularity": director_popularity,
    "Lead Actor Popularity": lead_actor_popularity,
    "Holiday Season": holiday_season,
    "Has Sequel": has_sequel,
    "Franchise": franchise
}])

# Add genre one-hot encoding
all_genres = ['Drama', 'Sport', 'Thriller', 'Fantasy', 'Animation', 'Crime', 'Sci-Fi', 'Action', 'Adventure', 'Romance', 'Documentary', 'Horror', 'Mystery', 'War', 'Family', 'Biography', 'Musical', 'Western', 'History', 'Comedy', 'Music']
for genre in all_genres:
    user_input[genre] = 1 if genre in genres else 0

# Prediction button
if st.button("Predict Revenue"):
    try:
        transformed_input = preprocessor.transform(user_input)
        predicted_revenue = model.predict(transformed_input)
        predicted_revenue = np.expm1(predicted_revenue)  
        if 'target_scaler.pkl' in os.listdir():  
            target_scaler = joblib.load("target_scaler.pkl")
            predicted_revenue = target_scaler.inverse_transform(predicted_revenue.reshape(-1, 1))
        st.success(f"🎥 Predicted Box Office Revenue: **${predicted_revenue[0]:,.2f}**")

    except Exception as e:
        st.error(f"Error: {str(e)} - Please check your inputs!")
