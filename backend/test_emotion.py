import pickle

# load trained model
model = pickle.load(open("emotion_model.pkl", "rb"))

# load vectorizer
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

print("Model and Vectorizer Loaded Successfully")

text = input("Enter your message: ")

# convert text into vector
text_vector = vectorizer.transform([text])

# predict emotion
prediction = model.predict(text_vector)

print("Predicted Emotion:", prediction[0])