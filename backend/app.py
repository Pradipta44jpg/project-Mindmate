import random
from flask_cors import CORS
from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
import anthropic
import base64

last_emotion = None

app = Flask(__name__)
CORS(app)

# ---------------- Load ML Model ----------------
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- NLP Setup ----------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- Anthropic Setup ----------------
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# ---------------- Home Route ----------------
@app.route("/")
def home():
    return "MindMate Backend Running"

# ---------------- Chat Emotion Detection ----------------
@app.route("/predict", methods=["POST"])
def predict_emotion():
    global last_emotion

    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    clean_text = preprocess_text(user_text)

    greetings = ["hi", "hello", "hey"]
    if clean_text in greetings:
        return jsonify({
            "emotion": "greeting",
            "reply": "Hello 👋 I'm MindMate. How are you feeling today?"
        })

    question_words = ["how", "what", "why", "are", "do", "can"]
    if any(word in user_text.lower().split() for word in question_words):
        return jsonify({
            "emotion": "question",
            "reply": "I'm doing well 😊 I'm here to listen to you."
        })

    emotion_keywords = {
        "sad": ["sad", "down", "unhappy", "cry", "depressed"],
        "happy": ["happy", "good", "great", "excited", "joy"],
        "angry": ["angry", "mad", "frustrated", "annoyed"],
        "lonely": ["lonely", "alone", "isolated"]
    }

    emotion_responses = {
        "sad": [
            "I’m really sorry you’re feeling this way. I’m here with you.",
            "It sounds heavy. You don’t have to face this alone.",
            "I’m listening. Take your time."
        ],
        "happy": [
            "That’s wonderful to hear 😊",
            "Your happiness matters.",
            "I love hearing this 😊"
        ],
        "angry": [
            "That sounds frustrating. I’m here.",
            "It’s okay to feel angry.",
            "Let’s slow down together."
        ],
        "lonely": [
            "You’re not alone. I’m here.",
            "I’m right here with you.",
            "You still matter 💙"
        ],
        "neutral": [
            "I’m here with you.",
            "Feel free to talk more."
        ]
    }

    fallback_responses = [
        "I’m listening.",
        "Please tell me more."
    ]

    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in clean_text.split():
                last_emotion = emotion
                return jsonify({
                    "emotion": emotion,
                    "reply": random.choice(emotion_responses[emotion])
                })

    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    previous_emotion = last_emotion
    last_emotion = prediction

    if previous_emotion in ["sad", "lonely"] and prediction == "neutral":
        reply = "Welcome back 💙 Earlier you sounded a bit low. How are you feeling now?"
    else:
        reply = random.choice(
            emotion_responses.get(prediction, fallback_responses)
        )

    return jsonify({
        "emotion": prediction,
        "reply": reply
    })

# ---------------- Face Emotion Detection ----------------
@app.route("/analyze-mood", methods=["POST"])
def analyze_mood():
    data = request.json
    image_base64 = data.get("image")

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        system="You are MindMate, a warm empathetic mental wellness companion.",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                {"type": "text", "text": "Detect my mood and respond kindly."}
            ]
        }]
    )

    text = response.content[0].text
    return jsonify({"reply": text})

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(debug=True)