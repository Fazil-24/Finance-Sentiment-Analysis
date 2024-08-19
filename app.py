import streamlit as st
import joblib
import re

# Load the saved model, vectorizer, and label encoder
model_filename = 'svm_sentiment_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'
le_filename = 'label_encoder.joblib'

# Load model, vectorizer, and label encoder
svm_model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)
le = joblib.load(le_filename)

# Function to clean and preprocess input text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Function to predict sentiment
def predict_sentiment(model, vectorizer, le, text):
    text_cleaned = clean_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)[0]
    sentiment = le.inverse_transform([prediction])[0]
    return sentiment

# Streamlit application
st.title('Sentiment Analysis with SVM')

user_input = st.text_area('Enter a statement:', '')
if st.button('Analyze Sentiment'):
    sentiment = predict_sentiment(svm_model, vectorizer, le, user_input)
    st.write('Sentiment:', sentiment)
