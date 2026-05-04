import streamlit as st
import pickle
import re

# =========================
# 🔥 PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================
# 🔥 LOAD MODEL + VECTORIZER
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# =========================
# 🔥 TEXT CLEANING FUNCTION (SAME AS TRAINING)
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text


# =========================
# 🎨 SIDEBAR
# =========================
st.sidebar.title("🧠 About Project")

st.sidebar.write("""
This Fake News Detection System uses Machine Learning 
to classify news articles as Fake or Real.
""")

st.sidebar.markdown("### ⚙️ Built using:")
st.sidebar.markdown("""
- TF-IDF
- Machine Learning
- Streamlit
""")

st.sidebar.success("🚀 Created by Muskan")

# =========================
# 📰 MAIN UI
# =========================
st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is **Fake or Real**")

# input box
user_input = st.text_area("✍️ Enter News Article Here")

# =========================
# 🔍 PREDICTION BUTTON
# =========================
if st.button("🔎 Analyze News"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first")

    else:
        # loading animation
        with st.spinner("Analyzing..."):

            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)

        # =========================
        # 🎯 RESULT DISPLAY
        # =========================
        if prediction == 0:
            st.error(f"🚨 This news is FAKE")
            st.write(f"Confidence: {round(prob[0][0] * 100, 2)}%")

        else:
            st.success(f"✅ This news is REAL")
            st.write(f"Confidence: {round(prob[0][1] * 100, 2)}%")