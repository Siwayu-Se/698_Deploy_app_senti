import streamlit as st
from transformers import pipeline

# กำหนด URL หรือเส้นทางของภาพพื้นหลัง
background_image_url = "https://images5.alphacoders.com/373/thumb-1920-373394.png"
text_color = "#FF0000"  # สีที่คุณต้องการ

# ใส่ CSS สำหรับพื้นหลังและสีตัวอักษร
st.markdown(
    f"""
<style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        height: 100vh;
        color: {text_color};
    }}
    h1, h2, h3, p, div {{
        color: {text_color} !important;
    }}
    .history {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }}
</style>
    """,
    unsafe_allow_html=True
)

# Load the sentiment analysis model
model_name = "poom-sci/WangchanBERTa-finetuned-sentiment"
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)

# Streamlit app
st.title("Thai Sentiment Analysis App")

# Input text
text_input = st.text_area("Enter Thai text for sentiment analysis", "ขอความเห็นหน่อย... ")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        # Analyze sentiment using the model
        results = sentiment_analyzer([text_input])

        # Extract sentiment and score
        sentiment = results[0]['label']
        score = results[0]['score']
        
        # Store the input text and result in history
        st.session_state.history.append((text_input, sentiment, score))

        # Display result as progress bars
        st.subheader("Sentiment Analysis Result:")
        if sentiment == 'pos':
            st.success(f"Positive Sentiment (Score: {score:.2f})")
            st.progress(score)
        elif sentiment == 'neg':
            st.error(f"Negative Sentiment (Score: {score:.2f})")
            st.progress(score)
        else:
            st.warning(f"Neutral Sentiment (Score: {score:.2f})")
            st.progress(score)

# Display search history
st.subheader("Search History:")
if st.session_state.history:
    for idx, (text, sentiment, score) in enumerate(st.session_state.history):
        st.markdown(f"""
        <div class="history">
            <strong>{idx + 1}. Text:</strong> {text}<br>
            <strong>Sentiment:</strong> {sentiment} <strong>(Score: {score:.2f})</strong>
        </div>
        """, unsafe_allow_html=True)
else:
    st.write("No search history yet.")

# Button to clear history
if st.button("Clear History"):
    st.session_state.history.clear()
    st.success("Search history cleared.")
