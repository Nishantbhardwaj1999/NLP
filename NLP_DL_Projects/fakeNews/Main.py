import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load the saved model and TF-IDF vectorizer
model_filename = 'rf_classifier_model.pkl'
model_data = joblib.load(model_filename)
rf_classifier = model_data['model']
tfidf_vectorizer = model_data['tfidf_vectorizer']

# Tokenization and preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return ' '.join(tokens)

def preprocess_input(text):
    cleaned_text = preprocess_text(text)  # Preprocess user input text
    input_tfidf = tfidf_vectorizer.transform([cleaned_text])  # Transform using loaded vectorizer
    return input_tfidf

def main():
    st.title('Fake News Detection')

    # Text input for user to enter news text
    input_text = st.text_area('Enter news text here:', '')

    if st.button('Check'):
        if input_text.strip() == '':
            st.warning('Please enter some text.')
        else:
            # Preprocess input text and transform using TF-IDF vectorizer
            input_tfidf = preprocess_input(input_text)
            
            # Make prediction using the loaded model
            prediction = rf_classifier.predict(input_tfidf)
            
            # Display prediction result
            if prediction[0] == 0:
                st.success('The news is REAL.')
            else:
                st.error('The news is FAKE.')

if __name__ == '__main__':
    main()
