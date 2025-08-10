# IMDB-Movie-Recommendation-System

An interactive **Streamlit-based recommendation system** that suggests movies based on a user's **plot description**, using **TF-IDF vectorization** and **Cosine Similarity** on preprocessed movie plot data.

---

## 🔍 Project Overview

### ✅ Data Source:
- 📄 Dataset: `imdb_2024_all_genres_movies_cleaned.csv` (scraped & cleaned in this project)
- Format: `.csv` with columns:  
  `genre`, `name`, `description`, `duration`, `rating`, `rating_count`

---

## ⚙️ NLP Preprocessing & Vectorization

### 📓 Notebook: `main.ipynb`

#### ✅ NLP Tasks Performed:
1. **Lowercasing & Stripping**  
   All text is converted to lowercase and stripped of extra spaces for uniformity.

2. **Tokenization**  
   Splits plot descriptions into individual words/tokens.

3. **Stopword Removal**  
   Removes common words like “the”, “and”, “is” that do not affect meaning.

4. **Lemmatization**  
   Converts words to their base form:  
   - *running* → *run*  
   - *mice* → *mouse*

5. **Punctuation Removal**  
   Filters out commas, periods, and other non-alphabetic characters.

---

### 📦 SpaCy Pipeline Optimization

```python
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
```

#### We disable:

parser → No need for dependency parsing (sentence structure), saves processing time.

ner → Named Entity Recognition is not required for similarity.

tagger → Part-of-speech tagging is unnecessary for our goal.

#### ⚡ Why?
Our task only needs tokenization + lemmatization + stopword removal, so disabling extra pipeline components speeds up processing significantly.

### 🔄 Vectorization:
TF-IDF Vectorization of cleaned movie plots

Computed Cosine Similarity Matrix for recommendations

### 📦 Saved Files:
movie_recommender.pkl — contains:

TF-IDF Vectorizer

TF-IDF Matrix

Cosine Similarity Matrix

Preprocessed DataFrame

## 🚀 Streamlit Application: app.py
### 🔧 Features:
#### 1️⃣ User Input:
Movie plot description (text area)

#### 2️⃣ Recommendation Engine:
Cleans and lemmatizes input

Transforms input into TF-IDF vector

Finds top 5 most similar movies using Cosine Similarity

#### 3️⃣ Output Display:
Movie name, genre, duration, rating & count (or "No ratings available")

Movie plot

Similarity score

## 💻 How to Run This Project
### 🧪 Step 1: Create Virtual Environment
```bash
python -m venv env_name
```
### ▶️ Step 2: Activate Environment
#### Windows
```bash
.\venv\Scripts\activate
```
#### macOS/Linux
```bash
source venv/bin/activate
```
### 📦 Step 3: Install Dependencies
```bash
pip install numpy pandas streamlit scikit-learn spacy
```
### 📥 Step 4: Download SpaCy Language Model
```bash
python -m spacy download en_core_web_sm
```
### 🚀 Step 5: Launch Streamlit App
```bash
streamlit run app.py
```
