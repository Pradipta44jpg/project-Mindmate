# train_emotion_model.py

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

print("=== Training script started ===")

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "dataset", "emotions.csv")

# Load dataset
data = pd.read_csv(dataset_path)

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)
print("Raw data:")
print(data.head())

# ==============================
# TEXT PREPROCESSING
# ==============================

# Download stopwords (only first time)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess_text)

print("\nAfter preprocessing:")
print(data[['text', 'clean_text']])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================
# TF-IDF + LOGISTIC REGRESSION
# ==============================

# Features and labels
X = data['clean_text']
y = data['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy: {accuracy * 100:.2f}%")

# Test with custom sentence
test_sentence = ["I an ok and relaxed"]
test_clean = [preprocess_text(test_sentence[0])]
test_vector = vectorizer.transform(test_clean)
prediction = model.predict(test_vector)

print(f"Test sentence: '{test_sentence[0]}'")
print(f"Predicted emotion: {prediction[0]}")

import pickle

# Save model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully")