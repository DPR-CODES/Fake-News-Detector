# 🔍 Fake News Detector

An AI-powered web application that detects fake news using Machine Learning and Natural Language Processing. Built with Streamlit, featuring an attractive UI with explainability features.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🎯 Real-time Prediction**: Instant fake news detection
- **📊 Confidence Scoring**: Probability-based predictions
- **🔬 Explainability**: 
  - Word cloud visualizations
  - Feature importance analysis
  - Suspicious keyword detection
- **💡 User Guidance**: Clear recommendations and warnings
- **🎨 Attractive UI**: Modern, responsive design
- **📈 Interactive Charts**: Visual analytics with Plotly

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fake-news-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Download NLTK data
- Train the Logistic Regression model
- Save `model.pkl` and `vectorizer.pkl`

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Project Structure

```
fake-news-detector/
│
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── model.pkl              # Trained model (generated)
├── vectorizer.pkl         # TF-IDF vectorizer (generated)
│
└── .streamlit/            # Streamlit config (optional)
    └── config.toml
```

## 🔧 How It Works

### Phase 1: Text Preprocessing
- Lowercase conversion
- URL/email removal
- Special character cleaning
- Stopword removal
- Lemmatization

### Phase 2: Feature Extraction
- TF-IDF Vectorization
- N-gram analysis (unigrams + bigrams)
- Maximum 5000 features

### Phase 3: Classification
- **Model**: Logistic Regression
- **Accuracy**: 85-95% (depending on dataset)
- **Output**: Binary classification (Fake/Real)

### Phase 4: Explainability
- Feature importance visualization
- Word cloud generation
- Suspicious keyword detection
- Confidence scoring

## 🌐 Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Push all files:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your repository
4. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.8+
5. Click "Deploy!"

### Step 3: Pre-train Model

**Important**: Before deploying, you need to commit the trained model files:

```bash
# Train the model locally
python train_model.py

# Add model files to git
git add model.pkl vectorizer.pkl
git commit -m "Add trained model files"
git push
```

Alternatively, you can use Streamlit Cloud's startup script to train on deployment.

## 📊 Using Real Datasets

To use actual fake news datasets, replace the sample data in `train_model.py`:

### Option 1: Kaggle Dataset

```python
# Download from: https://www.kaggle.com/c/fake-news/data
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
```

### Option 2: LIAR Dataset

```python
# Download from: https://huggingface.co/datasets/liar
# Contains labeled news with explanations
```

### Option 3: FakeNewsNet

```python
# Download from: https://github.com/KaiDMML/FakeNewsNet
# Includes social context data
```

## 🎨 UI Customization

### Change Theme Colors

Edit the gradient colors in `app.py`:

```python
# Find this section in the CSS
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

# Replace with your colors:
background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
```

### Add Logo

Replace the sidebar image URL:

```python
st.image("YOUR_LOGO_URL", width=100)
```

## 🔍 Model Performance

The model's performance depends on your training data:

| Metric | Score |
|--------|-------|
| Accuracy | 85-95% |
| Precision | 80-90% |
| Recall | 85-92% |
| F1-Score | 82-91% |

## 🛠️ Advanced Features

### Add More Models

In `train_model.py`, you can experiment with:

```python
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

# SVM
from sklearn.svm import SVC
model = SVC(probability=True)
```

### Add More Preprocessing

```python
# Add sentiment analysis
from textblob import TextBlob
sentiment = TextBlob(text).sentiment.polarity

# Add named entity recognition
import spacy
nlp = spacy.load('en_core_web_sm')
entities = [(ent.text, ent.label_) for ent in nlp(text).ents]
```

## 📱 API Integration

To add API endpoints:

```python
# Install FastAPI
pip install fastapi uvicorn

# Create api.py
from fastapi import FastAPI
import pickle

app = FastAPI()

@app.post("/predict")
def predict(text: str):
    # Your prediction logic
    return {"prediction": result, "confidence": confidence}
```

## 🐛 Troubleshooting

### Issue: NLTK Data Not Found

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Issue: Model File Not Found

Make sure to run `train_model.py` before starting the app:
```bash
python train_model.py
```

### Issue: Deployment Fails

Check that `model.pkl` and `vectorizer.pkl` are in your repository:
```bash
git add model.pkl vectorizer.pkl
git commit -m "Add model files"
git push
```

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Deep learning models (BERT, GPT)
- [ ] Fact-checking integration
- [ ] Source credibility scoring
- [ ] Social media integration
- [ ] Browser extension
- [ ] Mobile app
- [ ] Historical tracking
- [ ] Batch processing

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **NLTK** for NLP tools
- **Scikit-learn** for ML algorithms
- **Streamlit** for the amazing framework
- **Plotly** for interactive visualizations

## 📧 Contact

For questions or suggestions, please open an issue or reach out!

---

**⚠️ Disclaimer**: This tool is for educational purposes. Always verify news from multiple credible sources. No AI system is 100% accurate.

**Made with ❤️ and Python**
