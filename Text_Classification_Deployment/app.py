"""
Cyberbullying Text Detection — Streamlit Deployment
CT052-3-M-NLP | TP139069 Muhammad Mahdi & TP131138 Alieu Bojang

This app loads the best trained model (Linear SVC) and TF-IDF vectorizer,
accepts user input text, preprocesses it using the same pipeline as training,
and predicts whether the text contains cyberbullying content.
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Download required NLTK data ---
for resource in ['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

# --- Load model and vectorizer ---
@st.cache_resource
def load_model():
    """Load the saved model and TF-IDF vectorizer from pickle files."""
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, tfidf = load_model()

# --- Preprocessing (same pipeline as training notebook) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    """Convert NLTK POS tag to WordNet POS tag for better lemmatization."""
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def preprocess_text(text):
    """Clean and preprocess text using the same pipeline as training."""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # 3. Remove @mentions and # symbols
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    # 4. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. Tokenize
    tokens = word_tokenize(text)
    # 7. POS tagging
    tagged_tokens = nltk.pos_tag(tokens)
    # 8. Remove stopwords + POS-aware lemmatization
    tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
              for word, tag in tagged_tokens
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# --- Category display info ---
# Maps each prediction label to a user-friendly name and emoji
CATEGORY_INFO = {
    'not_cyberbullying': {'label': 'Not Cyberbullying', 'icon': '✅', 'color': 'green'},
    'age':              {'label': 'Age',               'icon': '👤', 'color': 'red'},
    'ethnicity':        {'label': 'Ethnicity',         'icon': '🌍', 'color': 'red'},
    'gender':           {'label': 'Gender',            'icon': '⚧',  'color': 'red'},
    'religion':         {'label': 'Religion',          'icon': '🕌', 'color': 'red'},
    'other_cyberbullying': {'label': 'Other Cyberbullying', 'icon': '⚠️', 'color': 'red'},
}

# ============================================================
# STREAMLIT UI
# ============================================================

# --- Page config ---
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This app detects cyberbullying in text using a **Linear SVC** model "
        "trained on TF-IDF features from the "
        "[Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) dataset."
    )

    st.markdown("---")
    st.markdown("### Model info")
    st.markdown("""
    | | |
    |---|---|
    | **Model** | Linear SVC (Tuned) |
    | **Accuracy** | 82.9% |
    | **F1 (macro)** | 0.824 |
    | **Features** | TF-IDF (10K) |
    | **Best params** | C=0.1, l2, squared_hinge |
    """)

    st.markdown("---")
    st.markdown("### Categories")
    for key, info in CATEGORY_INFO.items():
        st.markdown(f"{info['icon']} {info['label']}")

    st.markdown("---")
    st.caption("CT052-3-M-NLP · TP139069 & TP131138")

# --- Custom CSS for larger text ---
st.markdown("""
<style>
    /* Main title */
    h1[data-testid="stHeading"] { font-size: 2.8rem !important; }
    /* Description text */
    .main .block-container > div > div > div > div > div > .stMarkdown p { font-size: 1.2rem !important; }
    /* Text area label */
    .stTextArea label p { font-size: 1.3rem !important; font-weight: 600 !important; }
    /* Predict button */
    .stButton button { font-size: 1.4rem !important; padding: 0.7rem 1rem !important; }
</style>
""", unsafe_allow_html=True)

# --- Main content ---
st.title("🛡️ Cyberbullying Text Detector")
st.markdown("Enter any text below and the model will classify whether it contains cyberbullying content.")

# --- Text input ---
user_input = st.text_area(
    "Enter text to analyze",
    height=150,
    placeholder="Type or paste text here..."
)

# --- Predict button ---
if st.button("🔍 Predict", type="primary", use_container_width=True):
    if user_input.strip():
        # Preprocess the input text
        cleaned = preprocess_text(user_input)

        if cleaned.strip() == '':
            st.warning("After preprocessing, no meaningful words remain. Please enter more text.")
        else:
            # Vectorize and predict
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            # Get category info
            info = CATEGORY_INFO.get(prediction, {'label': prediction, 'icon': '❓', 'color': 'orange'})

            st.markdown("---")
            st.subheader("Prediction Result")

            # Display result with large, prominent styling
            if prediction == 'not_cyberbullying':
                st.markdown(
                    """
                    <div style="background-color: #d4edda; border: 2px solid #28a745;
                                border-radius: 12px; padding: 30px; text-align: center; margin: 10px 0;">
                        <h1 style="color: #155724; margin: 0;">✅ Not Cyberbullying</h1>
                        <p style="color: #155724; font-size: 18px; margin-top: 10px;">
                            The text does <b>not</b> appear to contain cyberbullying content.
                        </p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #f8d7da; border: 2px solid #dc3545;
                                border-radius: 12px; padding: 30px; text-align: center; margin: 10px 0;">
                        <h1 style="color: #721c24; margin: 0;">⚠️ Cyberbullying Detected</h1>
                        <p style="color: #721c24; font-size: 20px; margin-top: 15px;">
                            Category: <b>{info['icon']} {info['label']}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True
                )

            # Show preprocessing details in an expander
            with st.expander("View preprocessing details"):
                st.markdown(f"**Original text:** {user_input}")
                st.markdown(f"**Cleaned text:** {cleaned}")
                st.markdown(f"**Prediction:** `{prediction}`")
    else:
        st.warning("Please enter some text before clicking Predict.")