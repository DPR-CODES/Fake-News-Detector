# 🚀 Deployment Guide - Streamlit Cloud

This guide will help you deploy your Fake News Detector to Streamlit Cloud in minutes.

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (free at streamlit.io/cloud)
- Trained model files (model.pkl, vectorizer.pkl)

## 🔧 Step-by-Step Deployment

### 1. Prepare Your Project

First, train your model locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py
```

This creates `model.pkl` and `vectorizer.pkl` files.

### 2. Create GitHub Repository

#### Option A: Using GitHub Desktop or Web Interface

1. Go to github.com
2. Click "New Repository"
3. Name it: `fake-news-detector`
4. Make it Public
5. Click "Create repository"

#### Option B: Using Command Line

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Fake News Detector"

# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git
git branch -M main
git push -u origin main
```

### 3. Important: Add Model Files to Git

```bash
# Make sure model files are tracked
git add model.pkl vectorizer.pkl
git commit -m "Add trained model files"
git push
```

**⚠️ Note**: If your model files are too large (>100MB), see "Large Files" section below.

### 4. Deploy to Streamlit Cloud

1. **Go to**: https://streamlit.io/cloud
2. **Sign in** with GitHub
3. **Click**: "New app"
4. **Select**:
   - Repository: `YOUR_USERNAME/fake-news-detector`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click**: "Deploy!"

### 5. Wait for Deployment

- First deployment takes 2-5 minutes
- You'll see build logs
- Once complete, your app will be live!

## 🌐 Your App URL

Your app will be available at:
```
https://YOUR_USERNAME-fake-news-detector.streamlit.app
```

## 🔍 Handling Large Files

If your model files are >100MB, use Git LFS:

### Install Git LFS

```bash
# On Mac
brew install git-lfs

# On Ubuntu/Debian
sudo apt-get install git-lfs

# On Windows
# Download from: https://git-lfs.github.com/
```

### Setup Git LFS

```bash
# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add model.pkl vectorizer.pkl
git commit -m "Add model files with LFS"
git push
```

## 🛠️ Alternative: Train Model on Deployment

If you can't upload model files, train on first run:

### Create `setup.py`

```python
import subprocess
import sys

def setup():
    print("Setting up the application...")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Train model
    subprocess.check_call([sys.executable, "train_model.py"])
    
    print("Setup complete!")

if __name__ == "__main__":
    setup()
```

### Update `app.py`

Add this at the top:

```python
import os

# Train model if not exists
if not os.path.exists('model.pkl'):
    import subprocess
    subprocess.run(['python', 'train_model.py'])
```

## 📊 Using Real Dataset

### Option 1: Include in Repository

```bash
# Add dataset files
git add Fake.csv True.csv
git commit -m "Add datasets"
git push
```

### Option 2: Download on Startup

In `train_model.py`:

```python
import requests

def download_dataset():
    print("Downloading dataset...")
    
    # Example: Download from URL
    fake_url = "URL_TO_FAKE_CSV"
    true_url = "URL_TO_TRUE_CSV"
    
    # Download files
    # ... your download code ...
    
# Call in __main__
download_dataset()
```

### Option 3: Use Streamlit Secrets

For API keys or URLs:

1. Go to your app settings on Streamlit Cloud
2. Click "Secrets"
3. Add:

```toml
[datasets]
fake_news_url = "YOUR_URL"
true_news_url = "YOUR_URL"
```

In code:

```python
import streamlit as st

fake_url = st.secrets["datasets"]["fake_news_url"]
```

## 🔧 Environment Variables

### Set in Streamlit Cloud

1. Go to app settings
2. Click "Secrets"
3. Add environment variables:

```toml
[general]
debug_mode = false
max_features = 5000

[model]
random_state = 42
test_size = 0.2
```

### Use in code:

```python
import streamlit as st

if "model" in st.secrets:
    random_state = st.secrets["model"]["random_state"]
```

## 🐛 Troubleshooting

### Issue: Build Fails

**Check**:
- All files are in the repository
- `requirements.txt` is correct
- Python version compatibility

**Solution**:
```bash
# Test locally first
streamlit run app.py
```

### Issue: Model Not Found

**Check**:
- `model.pkl` is in the repository
- File path is correct in `app.py`

**Solution**:
```bash
# Verify files are tracked
git ls-files | grep .pkl
```

### Issue: Memory Error

**Cause**: Model too large

**Solutions**:
1. Use Git LFS (see above)
2. Reduce model size:
   ```python
   vectorizer = TfidfVectorizer(max_features=1000)  # Reduce from 5000
   ```
3. Use simpler model:
   ```python
   model = MultinomialNB()  # Smaller than Random Forest
   ```

### Issue: NLTK Data Download Fails

**Solution**: Add to `app.py`:

```python
import nltk
import os

@st.cache_resource
def download_nltk_data():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    nltk.data.path.append(nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
```

## 🎨 Custom Domain (Optional)

1. Buy a domain (e.g., from Namecheap, GoDaddy)
2. In Streamlit Cloud app settings, go to "Custom domain"
3. Follow instructions to add DNS records
4. Your app will be at: `www.yourfakenewsdetector.com`

## 📈 Monitor Your App

### View Analytics

1. Go to Streamlit Cloud dashboard
2. Click your app
3. View:
   - Usage statistics
   - Error logs
   - Performance metrics

### Setup Logging

In `app.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use in code
logger.info("User analyzed article")
logger.error("Prediction failed")
```

## 🔄 Update Your App

### Deploy Updates

```bash
# Make changes to your code
# ... edit files ...

# Commit and push
git add .
git commit -m "Update: improved UI"
git push
```

Streamlit Cloud auto-deploys on push! 🎉

### Rollback

If something breaks:

1. Go to Streamlit Cloud dashboard
2. Click "Reboot app"
3. Or revert git commit:
   ```bash
   git revert HEAD
   git push
   ```

## 🎯 Best Practices

1. **Test Locally First**
   ```bash
   streamlit run app.py
   ```

2. **Keep Secrets Secret**
   - Never commit API keys
   - Use Streamlit Secrets

3. **Monitor Performance**
   - Check app logs regularly
   - Monitor memory usage

4. **Version Control**
   - Tag releases:
     ```bash
     git tag -a v1.0 -m "First release"
     git push --tags
     ```

5. **Documentation**
   - Keep README updated
   - Add usage examples
   - Document API changes

## 📞 Get Help

- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Create issue in your repo

## ✅ Deployment Checklist

- [ ] Model trained locally
- [ ] All files committed to GitHub
- [ ] requirements.txt complete
- [ ] Model files in repository
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] App tested online
- [ ] Custom domain configured (optional)
- [ ] Analytics reviewed

---

**🎉 Congratulations!** Your Fake News Detector is now live!

Share your app URL and help fight misinformation! 🚀
