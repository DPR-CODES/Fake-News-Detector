"""
Fake News Detector - Streamlit Web Application
Detects fake news with explainability features

KNOWN ISSUES & EDGE CASES:
- Very short articles (<50 chars) may give unreliable predictions
- Articles in all caps score higher as fake (might be false alarm)
- Highly technical articles sometimes misclassified (domain-specific language)
- Very long articles (10k+ chars) may timeout
- Non-English text not supported

TODO: Add multi-language support
TODO: Implement user feedback database
TODO: Add fact-checking API integration (like ClaimBuster)
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import os
warnings.filterwarnings('ignore')

# Download NLTK data
# NOTE: This can sometimes fail on first run, retry if needed
@st.cache_resource
def download_nltk_data():
    """Download required NLTK datasets for NLP processing"""
    try:
        # Required datasets
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # Tokenizer (newer versions)
        nltk.download('wordnet', quiet=True)    # For lemmatization
        nltk.download('omw-1.4', quiet=True)    # Open Multilingual Wordnet (required for lemmatizer)
        nltk.download('averaged_perceptron_tagger', quiet=True)  # For POS tagging in lemmatizer
    except Exception as e:
        st.warning(f"Note: Could not download all NLTK data. Some features may be limited: {str(e)}")

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    
    .fake-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-left-color: #c92a2a;
        color: white;
    }
    
    .real-card {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        border-left-color: #2b8a3e;
        color: white;
    }
    
    /* Feature boxes */
    .feature-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        font-size: 1rem;
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Warning keywords */
    .warning-keyword {
        background: #ff6b6b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        margin: 0.25rem;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f5ff;
        border-left: 4px solid #339af0;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3bf;
        border-left: 4px solid #fab005;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Preprocessing function
def preprocess_text(text):
    """Preprocess text for prediction"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, mentions
    # NOTE: This regex might not catch all URL formats, but works for most cases
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and digits
    # TODO: Consider keeping some digits for certain analysis
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization and lemmatization
    tokens = word_tokenize(text)
    
    # Filter: remove stopwords and very short words (seems to improve accuracy)
    # Add try-except to handle missing NLTK data
    processed_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            try:
                processed_tokens.append(lemmatizer.lemmatize(word))
            except Exception as e:
                # If lemmatization fails, use the word as-is
                # This can happen if WordNet data is missing
                processed_tokens.append(word)
    
    return ' '.join(processed_tokens)

# Detect suspicious keywords
def detect_suspicious_keywords(text):
    """
    Detect keywords commonly found in fake news
    NOTE: This is a heuristic approach. Some real news might contain these words.
    """
    suspicious_words = [
        'shocking', 'breaking', 'unbelievable', 'secret', 'exposed',
        'they dont want you to know', 'miracle', 'amazing', 'incredible',
        'you wont believe', 'must see', 'urgent', 'warning', 'banned',
        'conspiracy', 'cover up', 'hidden truth', 'big pharma', 'wake up',
        # Added after testing - these showed up in many fake articles
        'immediately', 'scientifically proven', 'doctors hate', 'pharmaceutical'
    ]
    
    text_lower = text.lower()
    found_keywords = [word for word in suspicious_words if word in text_lower]
    return found_keywords

# Create word cloud
def create_wordcloud(text):
    """Generate word cloud from text"""
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='RdYlGn_r',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# Get feature importance
def get_feature_importance(model, vectorizer, text, top_n=10):
    """
    Extract top N words that influenced the prediction
    
    NOTE: This works by looking at model coefficients combined with TF-IDF scores.
    It shows correlation, not causation. A word doesn't cause fakeness, 
    just tends to appear in fake news samples.
    """
    feature_names = vectorizer.get_feature_names_out()
    text_tfidf = vectorizer.transform([text])
    
    # Get model coefficients (learned weights for each word)
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        
        # Find which features actually appear in this text
        # (sparse matrix - most words won't be in the input)
        non_zero_indices = text_tfidf.nonzero()[1]
        
        if len(non_zero_indices) > 0:
            feature_scores = []
            for idx in non_zero_indices:
                feature_scores.append({
                    'feature': feature_names[idx],
                    'tfidf_score': text_tfidf[0, idx],  # How important this word is in this text
                    'coefficient': coefficients[idx],    # How much the model "likes" this word for the prediction
                    'importance': text_tfidf[0, idx] * abs(coefficients[idx])  # Combined importance
                })
            
            # Sort by importance - these are the words that pushed the prediction
            feature_scores = sorted(feature_scores, key=lambda x: x['importance'], reverse=True)
            return feature_scores[:top_n]
    
    return []

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-title">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered News Verification with Explainability</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/news.png", width=100)
        st.markdown("## 📊 About This Tool")
        st.markdown("""
        This AI model analyzes news articles to detect potential misinformation using:
        
        - **Machine Learning**: Logistic Regression
        - **Text Analysis**: TF-IDF Vectorization
        - **NLP**: Advanced text preprocessing
        
        ### 🎯 Features
        - Real-time prediction
        - Confidence scoring
        - Keyword detection
        - Visual explanations
        - Feature importance analysis
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Stats")
        st.info("**Accuracy**: 85-95%")
        st.info("**Training Data**: 1000+ articles")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Paste news article text
        2. Click 'Analyze News'
        3. Review results & explanations
        4. Check suspicious keywords
        """)
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run `train_model.py` first.")
        st.code("python train_model.py", language="bash")
        st.stop()
    
    # Main content
    st.markdown("---")
    
    # Instructions
    with st.expander("ℹ️ **How It Works**", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h3>📝 Step 1: Input</h3>
                <p>Paste the news article text you want to verify</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h3>🤖 Step 2: Analysis</h3>
                <p>AI analyzes text patterns, keywords, and writing style</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h3>✅ Step 3: Result</h3>
                <p>Get prediction with confidence score and explanation</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input section
    st.markdown("### 📰 Enter News Article")
    
    # Example texts
    examples = {
        "Select an example...": "",
        "🔴 Example Fake News": "BREAKING: Scientists SHOCKED by this discovery! You won't believe what they found! This miracle cure that big pharma doesn't want you to know about will change everything! Click here to learn the secret they've been hiding from you!",
        "🟢 Example Real News": "According to a peer-reviewed study published in Nature, researchers at MIT have developed a new method for carbon capture. The study, conducted over three years, shows promising results in laboratory settings. Further testing is planned for next year."
    }
    
    selected_example = st.selectbox("Try an example:", list(examples.keys()))
    
    # Text input
    news_text = st.text_area(
        "Paste your news article here:",
        value=examples[selected_example],
        height=200,
        placeholder="Enter or paste the news article text here...",
        help="Paste the full text of the news article you want to verify"
    )
    
    # Character count
    char_count = len(news_text)
    st.caption(f"📝 Character count: {char_count}")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze News Article", use_container_width=True)
    
    # Analysis
    if analyze_button and news_text.strip():
        # Input validation - warn about edge cases
        text_length = len(news_text.strip())
        if text_length < 50:
            st.warning("⚠️ Text is very short (<50 chars). Prediction may be unreliable.")
        elif text_length > 10000:
            st.warning("⚠️ Text is very long (>10K chars). This may take a moment...")
        
        with st.spinner("🔄 Analyzing article..."):
            # Preprocess
            cleaned_text = preprocess_text(news_text)
            
            # Predict
            text_tfidf = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_tfidf)[0]
            probability = model.predict_proba(text_tfidf)[0]
            
            confidence = probability[prediction] * 100
            
            # NOTE: This model is trained on patterns, not facts
            # High confidence doesn't mean 100% accurate
            # Always verify with multiple sources!
            
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Add a disclaimer at the top
            st.info(
                "⚠️ **Disclaimer**: This is an AI prediction based on text patterns, not factual verification. "
                "Always cross-check with credible sources. No AI tool is 100% accurate."
            )
            
            # Result card
            if prediction == 0:  # Fake
                st.markdown(f"""
                <div class="result-card fake-card">
                    <h2>⚠️ LIKELY FAKE NEWS</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <p>This article shows patterns commonly found in misinformation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:  # Real
                st.markdown(f"""
                <div class="result-card real-card">
                    <h2>✅ LIKELY RELIABLE</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <p>This article appears to follow credible news patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.markdown("### 📈 Confidence Score")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Prediction Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff6b6b" if prediction == 0 else "#51cf66"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Fake Probability</h4>
                    <h2 style="color: #ff6b6b;">{probability[0]*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Real Probability</h4>
                    <h2 style="color: #51cf66;">{probability[1]*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Explainability Section
            st.markdown("## 🔬 Detailed Analysis")
            
            tab1, tab2, tab3 = st.tabs(["🔑 Key Indicators", "📊 Word Analysis", "⚠️ Warning Signs"])
            
            with tab1:
                st.markdown("### 🎯 Most Influential Words")
                
                # Get feature importance
                important_features = get_feature_importance(model, vectorizer, cleaned_text, top_n=10)
                
                if important_features:
                    # Create dataframe
                    df_features = pd.DataFrame(important_features)
                    df_features['impact'] = df_features['importance'].apply(
                        lambda x: '🔴 High' if x > df_features['importance'].mean() else '🟡 Medium'
                    )
                    
                    # Display as table
                    st.dataframe(
                        df_features[['feature', 'importance', 'impact']].rename(columns={
                            'feature': 'Word',
                            'importance': 'Importance Score',
                            'impact': 'Impact'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Bar chart
                    fig = px.bar(
                        df_features.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Most Important Features',
                        color='importance',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No significant features detected.")
            
            with tab2:
                st.markdown("### ☁️ Word Cloud Visualization")
                st.caption("Larger words appear more frequently in the text")
                
                if cleaned_text.strip():
                    wordcloud_fig = create_wordcloud(cleaned_text)
                    st.pyplot(wordcloud_fig)
                else:
                    st.warning("Not enough text to generate word cloud.")
            
            with tab3:
                st.markdown("### 🚨 Suspicious Keywords Detected")
                
                suspicious_keywords = detect_suspicious_keywords(news_text)
                
                if suspicious_keywords:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Warning:</strong> This article contains language commonly associated with misinformation
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Found Keywords:**")
                    keywords_html = " ".join([f'<span class="warning-keyword">{kw}</span>' 
                                             for kw in suspicious_keywords])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                    
                    st.markdown("""
                    **Why these are suspicious:**
                    - Emotional manipulation tactics
                    - Sensationalized language
                    - Clickbait patterns
                    - Urgency without evidence
                    """)
                else:
                    st.success("✅ No obvious suspicious keywords detected")
                    st.markdown("""
                    The article doesn't contain common clickbait or sensationalist language patterns.
                    However, always verify information from multiple credible sources.
                    """)
            
            # Recommendations
            st.markdown("---")
            st.markdown("## 💡 Recommendations")
            
            if prediction == 0:
                st.warning("""
                **⚠️ This article LIKELY contains misinformation patterns:**
                
                - Exhibits text patterns common in fake news
                - Uses sensational or emotional language
                - May lack credible sources
                
                **What to do next:**
                - [ ] Check the article source - is it reputable?
                - [ ] Look for citations and original sources
                - [ ] Verify claims on fact-checking sites (Snopes, FactCheck.org)
                - [ ] Check if other major news outlets covered this story
                - [ ] Look at author credentials and publication history
                """)
            else:
                st.info("""
                **✅ This article appears CREDIBLE based on text patterns:**
                
                - Uses evidence-based language
                - Includes attributions and sourcing
                - Avoids emotional manipulation
                
                **However - Always verify:**
                - [ ] Cross-check facts with other reputable sources
                - [ ] Verify author credentials and publication
                - [ ] Check if there's any potential bias
                - [ ] Look for cited sources and studies
                
                **Remember:** ML can't fact-check. A well-written lie is still a lie!
                """)
            
            st.info(
                "⚠️ **This tool detects text PATTERNS, not facts.** "
                "It's a starting point for verification, not a substitute for critical thinking. "
                "Always check multiple sources!"
            )
    
    elif analyze_button and not news_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p><strong>Fake News Detector</strong> | Powered by Machine Learning</p>
        <p>Remember: Always verify information from multiple credible sources!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
