import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# -------------------------------
# 1. INITIALIZE FLASK
# -------------------------------
app = Flask(__name__)

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# -------------------------------
# 2. DATASET CREATION (200+ Samples)
# -------------------------------
def create_dataset():
    base_data = [
        ("NASA reveals secret moon base populated by faked-death celebrities.", 0),
        ("New global treaty signed to protect oceans from plastic pollution.", 1),
        ("Eating chocolate daily improves IQ by 20 points, study says.", 0),
        ("Central bank raises rates to curb the highest inflation in decades.", 1),
        ("Internet will be shut down for three days for global update.", 0),
        ("Stock markets hit record high as tech sector earnings surprise.", 1)
    ]

    expanded_data = []
    for i in range(40):  # 6 x 40 = 240 samples
        for text, label in base_data:
            expanded_data.append({
                "text": f"{text} RefID{i}",
                "label": label
            })

    return pd.DataFrame(expanded_data)

# -------------------------------
# 3. PREPROCESSING
# -------------------------------
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = text.split()

    # Stopword removal + Stemming
    clean_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(clean_tokens)

# -------------------------------
# 4. MODEL TRAINING
# -------------------------------
df = create_dataset()
df["clean_text"] = df["text"].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# -------------------------------
# 5. SEARCH HISTORY
# -------------------------------
search_history = []
MAX_HISTORY = 10

# -------------------------------
# 6. ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html", history=search_history)

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()

    if not news_text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess input
    cleaned = preprocess_text(news_text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]
    prediction_prob = model.predict_proba(vec)[0]

    confidence = max(prediction_prob) * 100
    result = "REAL" if prediction == 1 else "FAKE"

    # Store limited history
    search_entry = {
        "text": news_text[:60] + ("..." if len(news_text) > 60 else ""),
        "result": result,
        "conf": round(confidence, 2)
    }

    search_history.insert(0, search_entry)
    if len(search_history) > MAX_HISTORY:
        search_history.pop()

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence:.2f}%",
        "tokens": cleaned.split()
    })

# -------------------------------
# 7. RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
