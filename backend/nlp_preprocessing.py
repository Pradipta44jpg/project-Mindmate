import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download needed data (run only first time)
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)

    clean_tokens = []
    for word in tokens:
        if word not in stop_words:
            clean_tokens.append(word)

    return clean_tokens


sentence = "I am feeling very lonely today 😢"
print(preprocess_text(sentence))
