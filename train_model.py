"""
Fake News Detection Model Training Script
Trains a machine learning model to detect fake news

Known issues:
- Small dataset leads to overfitting on samples
- Consider using real datasets from Kaggle for better performance
- Model training takes ~1-2 minutes on older laptops

TODO: Implement cross-validation
TODO: Try RandomForest and SVM models for comparison
TODO: Add hyperparameter tuning with GridSearch
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB  # Tried this, but LR works better
# from sklearn.ensemble import RandomForestClassifier  # Too slow, not worth the marginal improvement
# from sklearn.svm import SVC  # Tried SVM, but needs more tuning and slower to train
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Needed for newer NLTK versions
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Open Multilingual Wordnet (required for lemmatizer)
nltk.download('averaged_perceptron_tagger', quiet=True)  # For POS tagging in lemmatizer

class FakeNewsDetector:
    def __init__(self):
        # Current vectorizer config works well
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        # OTHER OPTIONS TRIED:
        # self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  # Slower, marginal improvement
        # self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))  # Trigrams add complexity, slower
        # self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)  # Removes rare words
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        # ALTERNATIVES CONSIDERED:
        # self.model = MultinomialNB()  # Faster but less accurate
        # self.model = RandomForestClassifier(n_estimators=100)  # Overkill, much slower
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Text preprocessing pipeline
        Applied to both training and prediction data
        
        Steps:
        1. Lowercase for consistency
        2. Remove URLs/emails/mentions
        3. Remove special chars and digits
        4. Tokenize, remove stopwords, lemmatize
        """
        if pd.isna(text):
            return ""
        
        text = text.lower()
        
        # Remove URLs - catches http, https, and www variants
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove @mentions and #hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        # NOTE: This might lose some info, but reduces noise significantly
        # ALTERNATIVE APPROACHES TRIED:
        # - Keep digits: text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Worse accuracy
        # - Remove only punctuation: text = re.sub(r'[^\w\s]', '', text)  # Worse accuracy
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words (len > 2 acts as filter)
        # Add error handling for missing NLTK data
        processed_tokens = []
        for word in tokens:
            if word not in self.stop_words and len(word) > 2:
                try:
                    processed_tokens.append(self.lemmatizer.lemmatize(word))
                except Exception:
                    # If lemmatization fails, use word as-is
                    processed_tokens.append(word)
        
        return ' '.join(processed_tokens)
    
    def load_and_prepare_data(self):
        """
        Load datasets and prepare for training
        
        Tries multiple approaches:
        1. Use fake_real_news.csv if it exists locally
        2. Try to download from GitHub
        3. Fall back to sample data if both fail
        """
        print("Loading datasets...")
        
        # Check if CSV file exists locally
        if os.path.exists('fake_real_news.csv'):
            print("Loading from fake_real_news.csv...")
            df = pd.read_csv('fake_real_news.csv')
            
            # Handle different column naming conventions
            # (different datasets call these different things)
            if 'label' not in df.columns:
                if 'target' in df.columns:
                    df['label'] = df['target']
                elif 'is_real' in df.columns:
                    df['label'] = df['is_real']
            
            # Combine title and text for richer features
            if 'title' in df.columns and 'text' in df.columns:
                df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            elif 'content' not in df.columns:
                if 'title' in df.columns:
                    df['content'] = df['title']
                elif 'text' in df.columns:
                    df['content'] = df['text']
        else:
            # Try to download dataset from GitHub
            # (This is a hack but allows the app to work without manual setup)
            print("Downloading fake news dataset from GitHub...")
            try:
                import urllib.request
                # Using ISOT dataset from GitHub
                url = "https://raw.githubusercontent.com/jainrachit/Fake-News-Detection/master/news.csv"
                urllib.request.urlretrieve(url, 'fake_real_news.csv')
                print("Dataset downloaded successfully!")
                df = pd.read_csv('fake_real_news.csv')
                
                # Map label column format
                if 'label' in df.columns:
                    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
                
                # Combine title and text
                if 'title' in df.columns and 'text' in df.columns:
                    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            except Exception as e:
                # If all else fails, use sample data
                print(f"Could not download dataset: {e}")
                print("Using sample data instead (WARNING: Very small dataset)...")
                df = self._create_sample_data()
        
        print(f"Dataset loaded: {len(df)} articles")
        print(f"Fake news: {len(df[df['label']==0])}")
        print(f"Real news: {len(df[df['label']==1])}")
        
        return df
    
    def _create_sample_data(self):
        """
        Create sample data for demonstration/testing
        
        NOTE: This is purely for demo purposes. The accuracy will be 100% 
        but meaningless. Real models need real data (1000+ articles minimum).
        
        This data is generated with obvious patterns that ML doesn't actually see
        in the real world. Use real datasets from Kaggle for actual work.
        """
        # Fake news samples - obviously fake with sensationalism
        fake_data = {
            'title': [
                'BREAKING: Shocking discovery!', 'You won\'t believe what happened!', 
                'URGENT: This will blow your mind', 'Scientists HATE this one trick',
                'Miracle cure discovered by accident', 'Government hiding truth',
                'Celebrity reveals shocking secret', 'This one weird trick doctors hate',
                'You haven\'t heard the REAL story', 'Fake news exposed!',
                'EXCLUSIVE: Never before seen footage', 'This will change EVERYTHING'
            ],
            'text': [
                'This is a fake sensational story with no evidence.' * 15,
                'Clickbait article with emotional manipulation and false claims.' * 15,
                'Unverified sources claim shocking things without proof.' * 15,
                'Misleading information designed to get clicks and shares.' * 15,
                'Debunked conspiracy theory resurfaced with added sensationalism.' * 15,
                'Anonymous sources claim extraordinary allegations without evidence.' * 15,
                'Out of context quotes used to create false narrative.' * 15,
                'Emotional appeals used instead of factual information.' * 15,
                'Manipulated statistics and misleading data visualization.' * 15,
                'Fake quotes attributed to real people.' * 15,
                'Doctored images presented as authentic.' * 15,
                'Sensational headlines with factually incorrect content.' * 15
            ],
            'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        true_data = {
            'title': [
                'Study shows climate change impact', 'Economic report released',
                'Healthcare policy update', 'Technology advancement reported',
                'University research breakthrough announced', 'Government official statement',
                'Market analysis by experts', 'Medical breakthrough confirmed',
                'Infrastructure project completion', 'Scientific discovery published',
                'International agreement reached', 'Corporate earnings report'
            ],
            'text': [
                'Peer-reviewed research indicates significant environmental changes.' * 15,
                'Official economic data shows quarterly growth in multiple sectors.' * 15,
                'Government announces new healthcare initiative based on expert advice.' * 15,
                'Tech company releases innovative product after years of development.' * 15,
                'Researchers at leading institution publish findings in peer-reviewed journal.' * 15,
                'Official statement released by authorized government spokesperson.' * 15,
                'Financial analysts provide detailed market analysis with supporting data.' * 15,
                'Medical establishment validates new treatment through clinical trials.' * 15,
                'Major infrastructure project completed on schedule and within budget.' * 15,
                'Scientists publish discoveries in reputable academic journals.' * 15,
                'Multiple nations reach consensus on international policy matters.' * 15,
                'Company releases quarterly results audited by independent firms.' * 15
            ],
            'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        
        fake_df = pd.DataFrame(fake_data)
        true_df = pd.DataFrame(true_data)
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df['content'] = df['title'] + ' ' + df['text']
        
        return df
    
    def train(self):
        """
        Complete training pipeline
        
        NOTE: With sample data (8 articles), accuracy will be 100% but meaningless.
        The model needs at least 1000+ articles to learn real patterns.
        """
        # Load data
        df = self.load_and_prepare_data()
        
        # Preprocess text - this is where most of the cleanup happens
        print("\nPreprocessing text...")
        df['cleaned_text'] = df['content'].apply(self.preprocess_text)
        
        # Split data - standard 80/20 split
        # stratify ensures balanced fake/real in both train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], 
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        # TODO: Add cross-validation for better accuracy estimates
        
        # Feature extraction with TF-IDF
        # This converts text into numerical features that ML models can understand
        print("\nExtracting features with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model - Logistic Regression is simple, fast, and works well for this
        # Tried Naive Bayes but LogReg had better accuracy on our data
        print("\nTraining Logistic Regression model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Fake News', 'Real News']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model and vectorizer for app to use
        print("\nSaving model and vectorizer...")
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("✓ Model saved successfully!")
        
        return accuracy
    
    def predict(self, text):
        """
        Predict if news is fake or real
        """
        cleaned_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return prediction, probability

if __name__ == "__main__":
    print("🚀 Starting Fake News Detection Model Training\n")
    
    detector = FakeNewsDetector()
    accuracy = detector.train()
    
    print(f"\n✅ Training completed with {accuracy*100:.2f}% accuracy!")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")
