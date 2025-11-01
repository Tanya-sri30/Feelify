# app.py
import streamlit as st
import joblib

# -------------------------------
# 1️⃣ Load the Trained Model, Vectorizer, and Label Encoder
# -------------------------------
# Make sure these files exist in your project folder
model = joblib.load("emotion_model.pkl")           # Trained ML model
vectorizer = joblib.load("tfidf_vectorizer.pkl")   # TF-IDF vectorizer used during training
label_encoder = joblib.load("label_encoder.pkl")   # Label encoder used to map numbers ↔ emotions

# -------------------------------
# 2️⃣ Streamlit App Title and Description
# -------------------------------
st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="centered")
# Optional custom CSS for better visuals
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            border: 2px solid #4A90E2;
            border-radius: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
        .emotion-card {
            background: #ffffffcc;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎭 Emotion Detection from Text")
st.write("This app predicts the **emotion** hidden behind your text using NLP & ML!")

# -------------------------------
# 3️⃣ Text Input from User
# -------------------------------
user_input = st.text_area(
    "Enter a sentence or paragraph:",
    height=150,
    placeholder="Type something like — I enjoy solving code problems even when I'm tired."
)

# -------------------------------
# 4️⃣ Predict Emotion on Button Click
# -------------------------------
if st.button("🔍 Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform text using saved TF-IDF vectorizer
        transformed_input = vectorizer.transform([user_input])

        # Predict encoded emotion label
        predicted_label = model.predict(transformed_input)[0]

        # Decode label to emotion text
        predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

        # Add fun emoji mapping
        emoji_dict = {
            'joy': '😊',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'love': '❤️',
            'surprise': '😲'
        }
        emoji = emoji_dict.get(predicted_emotion.lower(), '🙂')

        # Display result
        st.success(f"Predicted Emotion: **{predicted_emotion.capitalize()}** {emoji}")

# -------------------------------
# 5️⃣ Optional: Footer
# -------------------------------
st.markdown("---")
st.caption("Built with 💬 Streamlit | NLP Emotion Classifier Model")
