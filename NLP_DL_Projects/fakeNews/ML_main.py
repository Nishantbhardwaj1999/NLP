import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess the dataset
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercasing and removing stopwords/punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def train_and_save_model(file_path):
    # Load your dataset
    df = pd.read_csv(file_path)

    # Preprocess text column
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

    # Vectorize text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_tfidf, y_train)

    # Predict on the test set
    y_pred = rf_classifier.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Display classification report
    print(classification_report(y_test, y_pred))

    # Define the filename for saving the model
    model_filename = 'rf_classifier_model.pkl'

    # Save the trained model and TF-IDF vectorizer to a file
    model_data = {
        'model': rf_classifier,
        'tfidf_vectorizer': tfidf_vectorizer
    }
    joblib.dump(model_data, model_filename)

# Specify the file path of your dataset
file_path = "D:\\NLP\\NLP_DL_Projects\\fakeNews\\DataSet\\fakenews.csv"

# Train and save the model
train_and_save_model(file_path)
