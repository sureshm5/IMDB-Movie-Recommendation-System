import streamlit as st
import pickle
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed data
@st.cache_data
def load_data():
    with open('movie_recommender.pkl', 'rb') as f:
        return pickle.load(f)
data = load_data()
tfidf, tfidf_matrix, cosine_sim, df = data['tfidf'], data['tfidf_matrix'], data['cosine_sim'], data['movies']

# NLP processing function
def process_text(text):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Streamlit UI
st.title("🎬 Movie Recommendation Engine")
user_input = st.text_area("Describe a movie plot:")

if user_input:
    # Process input
    processed_input = process_text(user_input.lower().strip())
    input_vector = tfidf.transform([processed_input])
    
    # Get similarities
    sim_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-5:][::-1]  # Top 5 matches
    
    # Display results
    st.write("### Top Recommendations")
    for idx in top_indices:
        rating = df.iloc[idx]['rating']
        rating_count = df.iloc[idx]['rating_count']

        # Replace NaN with message
        if pd.isna(rating) or pd.isna(rating_count):
            rating_text = "No ratings available"
        else:
            rating_text = f"⭐{rating} {rating_count}"

        st.write(f"#### {df.iloc[idx]['name']}")
        st.write(f"**Genre**: {df.iloc[idx]['genre']} | "
                 f"**Duration**: {df.iloc[idx]['duration']} | "
                 f"**Rating**: {rating_text}")
        st.write(f"**Plot**: {df.iloc[idx]['description']}")
        st.write(f"**Similarity Score**: `{sim_scores[idx]:.2f}`")
        st.write("---")