import random
from flask_cors import CORS
from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

# -------------------- Flask App --------------------
app = Flask(__name__)
CORS(app)

# -------------------- Load Model & Vectorizer --------------------
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------- NLP Setup --------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------- Home Route --------------------
@app.route("/")
def home():
    return "MindMate Backend Running"

# -------------------- Prediction Route --------------------
@app.route("/predict", methods=["POST"])
def predict_emotion():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    clean_text = preprocess_text(user_text)

    # -------- Emotion Keywords (Rule-based support) --------
    emotion_keywords = {
        "sad": ["sad", "down", "unhappy", "cry", "depressed"],
        "happy": ["happy", "good", "great", "excited", "joy"],
        "angry": ["angry", "mad", "frustrated", "annoyed"],
        "lonely": ["lonely", "alone", "isolated"]
    }

    # -------- Empathetic Responses --------
    emotion_responses = {
        "sad": [
            "I’m really sorry you’re feeling this way. I’m here with you.",
            "It sounds heavy. You don’t have to face this alone.",
            "I’m listening. Take your time and share what you feel."
        ],
        "happy": [
            "That’s wonderful to hear 😊 I’m glad you’re feeling good.",
            "Your happiness matters. Enjoy this moment.",
            "I love hearing this. Keep smiling 😊"
        ],
        "angry": [
            "That sounds frustrating. Let’s take a deep breath together.",
            "It’s okay to feel angry. I’m here to listen.",
            "I understand this can be overwhelming."
        ],
        "lonely": [
            "You’re not alone. I’m right here with you.",
            "Even when it feels quiet, you still matter.",
            "I’m here to keep you company."
        ],
        "neutral": [
            "Thanks for sharing. How has your day been so far?",
            "I’m here with you. Feel free to talk more.",
            "It’s okay to feel neutral sometimes."
        ]
    }

    fallback_responses = [
        "I’m here with you. Please tell me more.",
        "Your feelings matter, even if they’re hard to explain.",
        "I’m listening. Take your time.",
        "It’s okay if you’re not sure how you feel."
    ]

    # -------- Keyword-based override (short but emotional inputs) --------
    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in clean_text.split():
                return jsonify({
                    "emotion": emotion,
                    "reply": random.choice(emotion_responses[emotion])
                })

    # -------- Truly unclear input --------
    if len(clean_text.split()) < 3:
        return jsonify({
            "emotion": "neutral",
            "reply": random.choice(emotion_responses["neutral"])
        })

    # -------- ML Prediction --------
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    reply = random.choice(
        emotion_responses.get(prediction, fallback_responses)
    )

    return jsonify({
        "emotion": prediction,
        "reply": reply
    })

# -------------------- Run Server --------------------
if __name__ == "__main__":
    app.run(debug=True)

