"""
Fake News Detection Model Training Script
This script trains a machine learning model to detect fake news
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_and_prepare_data(self):
        """
        Load datasets and prepare for training
        """
        print("Loading datasets...")
        
        # Check if CSV file exists
        import os
        if os.path.exists('fake_real_news.csv'):
            print("Loading from fake_real_news.csv...")
            df = pd.read_csv('fake_real_news.csv')
            
            # Ensure label column exists (0=fake, 1=real)
            if 'label' not in df.columns:
                if 'target' in df.columns:
                    df['label'] = df['target']
                elif 'is_real' in df.columns:
                    df['label'] = df['is_real']
            
            # Combine title and text if separate columns exist
            if 'title' in df.columns and 'text' in df.columns:
                df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            elif 'content' not in df.columns:
                if 'title' in df.columns:
                    df['content'] = df['title']
                elif 'text' in df.columns:
                    df['content'] = df['text']
        else:
            # Try to download ISOT dataset from GitHub
            print("Downloading fake news dataset from GitHub...")
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/jainrachit/Fake-News-Detection/master/news.csv"
                urllib.request.urlretrieve(url, 'fake_real_news.csv')
                print("Dataset downloaded successfully!")
                df = pd.read_csv('fake_real_news.csv')
                
                # Map label column
                if 'label' in df.columns:
                    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
                
                # Use title + text as content
                if 'title' in df.columns and 'text' in df.columns:
                    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            except Exception as e:
                print(f"Could not download dataset: {e}")
                print("Using sample data instead...")
                df = self._create_sample_data()
        
        print(f"Dataset loaded: {len(df)} articles")
        print(f"Fake news: {len(df[df['label']==0])}")
        print(f"Real news: {len(df[df['label']==1])}")
        
        return df
    
    def _create_sample_data(self):
        """
        Create comprehensive sample data for demonstration
        """
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
        """
        # Load data
        df = self.load_and_prepare_data()
        
        # Preprocess text
        print("\nPreprocessing text...")
        df['cleaned_text'] = df['content'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], 
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        # Feature extraction
        print("\nExtracting features with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        print("\nTraining Logistic Regression model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
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
        
        # Save model and vectorizer
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
