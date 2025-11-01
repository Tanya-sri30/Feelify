🎭 Feelify — Emotion Detection from Text








🧠 Overview

Feelify is an NLP-powered web app that detects the emotional tone hidden within text.
It uses a Machine Learning model trained on labeled emotion data and deployed using Streamlit.

⚡ Built with a learning motive — Feelify may not be 100% accurate, but it demonstrates the full NLP → ML → Deployment workflow.

🔗 Live Demo: feelify-30.streamlit.app

📦 Repository: github.com/Tanya-sri30/Feelify

🎯 What It Does

Feelify analyzes input text and predicts emotions such as:

joy, sadness, anger, fear, love, surprise

Using TF-IDF Vectorization and Logistic Regression, it converts plain sentences into numerical features and classifies the dominant emotion.

⚙️ Features

🧩 Text Preprocessing — tokenization, stopword removal, lemmatization

💬 TF-IDF Vectorization for numerical feature extraction

🤖 Emotion detection using Logistic Regression & Naive Bayes

🌐 Interactive web UI built with Streamlit

📊 Model, vectorizer & label encoder stored with joblib

☁️ Deployed live on Streamlit Cloud

🧩 Tech Stack
Category	Technology
Language	Python 🐍
NLP Libraries	NLTK, scikit-learn
Model	Logistic Regression
Frontend / UI	Streamlit
Deployment	Streamlit Cloud
Version Control	Git & GitHub
🧾 Project Structure
Feelify/
│
├── frontend.py            # Streamlit app (main interface)
├── emotion_model.pkl      # Trained ML model
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── label_encoder.pkl      # Label encoder
├── train.txt              # Training dataset
├── requirements.txt       # Required dependencies
└── README.md              # Project documentation

🧪 Model Performance
Model	Accuracy
Naive Bayes	77.3%
Logistic Regression	87.1% ✅