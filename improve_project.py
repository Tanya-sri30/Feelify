# ===============================
# 📘 Improved Emotion Detection Model
# ===============================

# 1️⃣ Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 2️⃣ Load dataset
df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotions"])
print(" Data Loaded Successfully")
print(df.head())
print(df.isnull().sum())

# 3️⃣ Encode emotion labels to numbers
le = LabelEncoder()
df["emotions"] = le.fit_transform(df["emotions"])

# 4️⃣ Text Preprocessing

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove numbers
    text = ''.join([ch for ch in text if not ch.isdigit()])
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned_tokens)

print("Cleaning text ... (this might take a bit)")
df["clean_text"] = df["text"].apply(clean_text)

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["emotions"], test_size=0.2, random_state=42)

# 6️⃣ Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # using bigrams too improves accuracy
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 7️⃣ Train Models
# --- Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)
print("\n Naive Bayes Accuracy:", nb_acc)
print(classification_report(y_test, nb_pred))

# --- Logistic Regression (usually better for text classification) ---
logistic_model = LogisticRegression(max_iter=2000, solver='lbfgs')
logistic_model.fit(X_train_tfidf, y_train)
log_pred = logistic_model.predict(X_test_tfidf)
log_acc = accuracy_score(y_test, log_pred)
print("\n Logistic Regression Accuracy:", log_acc)
print(classification_report(y_test, log_pred))

# 8️⃣ Save the best model
if log_acc > nb_acc:
    joblib.dump(logistic_model, "emotion_model.pkl")
    print(" Logistic Regression model saved as emotion_model.pkl")
else:
    joblib.dump(nb_model, "emotion_model.pkl")
    print("Naive Bayes model saved as emotion_model.pkl")

joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Vectorizer and Label Encoder Saved Successfully!")
print(f"Final Model Accuracy: {max(log_acc, nb_acc):.4f}")
