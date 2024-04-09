import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the TF-IDF vectorizer and Logistic Regression model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

def predict_category(resume_text):
    cleaned_text = preprocess_text(resume_text)
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])
    category = classifier.predict(text_vectorized)[0]
    return category

def main():
    st.title('Resume Category Classifier')
    uploaded_file = st.file_uploader("Upload a resume", type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            # Attempt to read and decode the uploaded file
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8', errors='replace')
            
            st.text_area("Uploaded Resume", resume_text)

            if st.button("Classify"):
                category = predict_category(resume_text)
                st.success(f"Predicted Category: {category}")

        except Exception as e:
            st.error(f"Error: {e}. Please try again with a different file.")

if __name__ == "__main__":
    main()
