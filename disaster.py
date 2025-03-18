import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.corpus import stopwords
import string
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
dataset_path = "larger_disaster_news_dataset.csv"
df = pd.read_csv(dataset_path).dropna()
if "news_text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'news_text' and 'label' columns.")
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df["news_text"] = df["news_text"].apply(preprocess_text)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(df["news_text"], df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Training the model, please wait...")
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_data=(X_test_tfidf, y_test))
model.save("fake_news_model.h5")
print("Model training complete and saved as 'fake_news_model.h5'.")
model = tf.keras.models.load_model("fake_news_model.h5")
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    probability = model.predict(vectorized_text)[0][0]
    real_prob = round(probability * 100, 2)
    fake_prob = round((1 - probability) * 100, 2)
    prediction = "REAL" if real_prob > fake_prob else "FAKE"

    return jsonify({
        "prediction": prediction,
        "credibility_score": f"{max(real_prob, fake_prob)}%"
    })

if __name__ == '__main__':
    print("\n===== Disaster Fake News Detection =====")
    news_text = input("Enter the news article text: ")

    if news_text.strip():
        processed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        
        probability = model.predict(vectorized_text)[0][0]
        real_prob = round(probability * 100, 2)
        fake_prob = round((1 - probability) * 100, 2)
        prediction = "REAL" if real_prob > fake_prob else "FAKE"

        print("\n===== RESULT =====")
        print(f"📰 Input News: {news_text}")
        print(f"🔍 Prediction: {prediction}")
        print(f"📊 Credibility Score: {max(real_prob, fake_prob)}%")
    else:
        print("No input provided. Exiting...")
    
    app.run(debug=True, threaded=False)