import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

#############################################  Step 1: Pre-process the Data #####################################

# 1. Loading Data
df = pd.read_csv('archive/data.csv')
df = df.dropna()

# Cleaning and sanitizing
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
    text = text.lower()  
    return text

df['Sentence'] = df['Sentence'].apply(clean_text)

# 3. Converting to numerical labels 
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df['Sentiment'])

# 4. Splitting data
X = df['Sentence']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Extraction 
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#############################################  Step 2: Build the SVM model #####################################

# Define SVM model
svm_model = SVC(kernel='linear', probability=True)  # Using linear kernel

# Train SVM model
svm_model.fit(X_train_tfidf, y_train)

#############################################  Step 3: Evaluate the model #####################################

# Predict on test data
y_pred = svm_model.predict(X_test_tfidf)

# Model Performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

#############################################  Step 4: Save the model #####################################

# Save the trained model
model_filename = 'svm_sentiment_model.joblib'
joblib.dump(svm_model, model_filename)
print(f'Model saved to {model_filename}')

# Save the vectorizer
vectorizer_filename = 'tfidf_vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)
print(f'Vectorizer saved to {vectorizer_filename}')

# Save the label encoder
le_filename = 'label_encoder.joblib'
joblib.dump(le, le_filename)
print(f'Label Encoder saved to {le_filename}')
