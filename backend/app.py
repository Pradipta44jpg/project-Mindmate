import random
from flask_cors import CORS
from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
import anthropic

# ---------------- GLOBAL ----------------
last_emotion = None

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL ----------------
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- NLP SETUP ----------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- AI SETUP ----------------
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# ---------------- HOME ----------------
@app.route("/")
def home():
    return "MindMate Backend Running 🚀"

# ---------------- CHAT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict_emotion():
    global last_emotion

    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    clean_text = preprocess_text(user_text)

    # -------- Greeting --------
    greetings = ["hi", "hello", "hey"]
    if clean_text in greetings:
        return jsonify({
            "emotion": "greeting",
            "reply": "Hello 👋 I'm MindMate. How are you feeling today?"
        })

    # -------- Simple question --------
    question_words = ["how", "what", "why", "are", "do", "can"]
    if any(word in user_text.lower().split() for word in question_words):
        return jsonify({
            "emotion": "question",
            "reply": "I'm doing well 😊 I'm here to listen to you."
        })

    # -------- Keyword detection --------
    emotion_keywords = {
        "sad": ["sad", "down", "unhappy", "cry", "depressed"],
        "happy": ["happy", "good", "great", "excited", "joy"],
        "angry": ["angry", "mad", "frustrated", "annoyed"],
        "lonely": ["lonely", "alone", "isolated"]
    }

    emotion_responses = {
        "sad": [
            "I’m really sorry you’re feeling this way 💙",
            "I’m here with you, take your time",
            "You don’t have to go through this alone"
        ],
        "happy": [
            "That’s wonderful 😊",
            "I love hearing that!",
            "Your happiness matters 💛"
        ],
        "angry": [
            "Take a deep breath 😌",
            "That sounds frustrating",
            "I’m here, let it out"
        ],
        "lonely": [
            "You’re not alone 💙",
            "I’m here with you",
            "Let’s talk 😊"
        ],
        "neutral": [
            "I’m listening 😊",
            "Tell me more",
            "I’m here for you"
        ]
    }

    fallback_responses = [
        "I’m listening 😊",
        "Tell me more",
        "I understand"
    ]

    # Check keywords first
    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in clean_text.split():
                last_emotion = emotion
                return jsonify({
                    "emotion": emotion,
                    "reply": random.choice(emotion_responses[emotion])
                })

    # -------- ML Prediction --------
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    previous_emotion = last_emotion
    last_emotion = prediction

    # -------- Memory Logic --------
    if previous_emotion in ["sad", "lonely"] and prediction == "neutral":
        reply = "Welcome back 💙 Earlier you seemed low. How are you now?"

    else:
        try:
            # -------- AI RESPONSE --------
            ai_response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
User said: {user_text}
Detected emotion: {prediction}
Previous emotion: {previous_emotion}

Act like a caring emotional support friend.
Talk gently, short, and human.
"""
                    }
                ]
            )

            reply = ai_response.content[0].text

        except Exception as e:
            print("AI Error:", e)
            reply = random.choice(
                emotion_responses.get(prediction, fallback_responses)
            )

    # -------- FINAL RESPONSE --------
    return jsonify({
        "emotion": prediction,
        "reply": reply
    })


# ---------------- FACE ROUTE ----------------
@app.route("/analyze-mood", methods=["POST"])
def analyze_mood():
    data = request.json
    image_base64 = data.get("image")

    if not image_base64:
        return jsonify({
            "reply": "I couldn't see your face clearly."
        })

    print("Image received from frontend")

    return jsonify({
        "reply": "Hello 😊 I can see you. How are you feeling today?"
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)