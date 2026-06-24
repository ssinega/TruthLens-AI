```
████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗██╗     ███████╗███╗  ██╗███████╗
   ██╔══╝ ██╔══██╗██║   ██║╚══██╔══╝██║  ██║██║     ██╔════╝████╗ ██║██╔════╝
   ██║    ██████╔╝██║   ██║   ██║   ███████║██║     █████╗  ██╔██╗██║███████╗
   ██║    ██╔══██╗██║   ██║   ██║   ██╔══██║██║     ██╔══╝  ██║╚████║╚════██║
   ██║    ██║  ██║╚██████╔╝   ██║   ██║  ██║███████╗███████╗██║ ╚███║███████║
   ╚═╝    ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚══╝╚══════╝
                        "See Through the Noise"
```

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![IBM Edunet](https://img.shields.io/badge/IBM-Edunet%202025-054ADA?style=for-the-badge&logo=ibm)](https://ibm.com)

**AI-powered fake news detection tool for students and young news consumers.**

</div>

---

## 📌 Problem Statement (IBM Edunet)

> In an era of social media saturation, students encounter hundreds of news articles daily with limited ability to evaluate their credibility. Misinformation spreads 6x faster than true news. TruthLens addresses this by providing an accessible, explainable AI tool that teaches students *how* to identify fake news while detecting it automatically.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📝 **Multi-Input Support** | Paste text, enter URL (auto-scraped), or upload image (OCR) |
| 📊 **TruthScore™ Gauge** | Animated 0-100 credibility score with color-coded verdict |
| 🤖 **9 AI Models** | Supervised + Unsupervised + Deep Learning ensemble |
| 🧠 **XAI Explainability** | LIME word-level highlights — understand *why* AI flagged content |
| 🗺️ **Cluster Explorer** | K-Means PCA scatter plot placing your article in news landscape |
| 📉 **Style Radar Chart** | 6-axis writing style profile (sensationalism, bias, citations...) |
| ☁️ **Red-Flag Wordcloud** | Top 10 suspicious words highlighted visually |
| 🎓 **Learning Module** | Algorithm explanations + fact-check tips + credibility checklist |
| 🏆 **Gamification** | TruthHunter points, badge system, session leaderboard |
| 🌐 **Language Detection** | Warning for non-English articles |
| 🤖 **Claude AI Brief** | Anthropic Claude generates a 3-bullet credibility brief per article |
| 💬 **Ask Claude** | Interactive Q&A — ask Claude anything about the article |

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TruthLens — Data Flow                           │
└─────────────────────────────────────────────────────────────────────┘

        Input Sources
       ┌──────────────────────────────────┐
       │  Text │  URL (scraped)  │  Image │
       └──────────────┬───────────────────┘
                      │
              ┌───────▼────────┐
              │ TextPreprocessor│  ← NLTK tokenize, lemmatize, clean
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │FeatureExtractor│  ← TF-IDF, VADER, Linguistic features
              └──┬─────────────┘
                 │
        ┌────────┴──────────────────────────────────────────────┐
        │                   ML Pipeline                          │
        │  ┌──────────────────────────────────────────────────┐ │
        │  │ SUPERVISED                                        │ │
        │  │  1. Logistic Regression ──────────────────────┐  │ │
        │  │  2. Random Forest ────────────────────────────┤  │ │
        │  │  3. Passive Aggressive Classifier ────────────┤──►Soft│
        │  │  4. LinearSVC ────────────────────────────────┘  │Voting│
        │  │  5. DistilBERT (optional) ────────────────────►Ensemble│
        │  └──────────────────────────────────────────────────┘ │
        │  ┌──────────────────────────────────────────────────┐ │
        │  │ UNSUPERVISED                                      │ │
        │  │  6. K-Means Clustering (Cluster Map) ──────────► │ │
        │  │  7. TF-IDF + Cosine Similarity ────────────────► │ │
        │  │  8. Isolation Forest (Anomaly) ────────────────► │ │
        │  └──────────────────────────────────────────────────┘ │
        └────────────────────────────┬──────────────────────────┘
                                     │
                      ┌──────────────▼───────────────────┐
                      │        CredibilityScorer          │
                      │  Linguistic + Sentiment + Source  │
                      │  Citation + Headline Consistency  │
                      │  + Clickbait = TruthScore™ (0-100)│
                      └──────────────┬───────────────────┘
                                     │
                      ┌──────────────▼───────────────────┐
                      │     Streamlit Dashboard           │
                      │  Gauge │ Radar │ LIME │ Clusters  │
                      │  Breakdown │ WordCloud │ PDF      │
                      └───────────────────────────────────┘
```

---

## 🤖 ML Algorithms

| # | Algorithm | Type | Purpose | Typical Accuracy |
|---|-----------|------|---------|-----------------|
| 1 | **Logistic Regression** (TF-IDF) | Supervised | Baseline classifier — linear word pattern matching | ~94% |
| 2 | **Random Forest** | Supervised (Ensemble) | Decision tree ensemble — robust to overfitting | ~92% |
| 3 | **Passive Aggressive Classifier** | Supervised (Online) | Fast online learning — ideal for streaming news | ~91% |
| 4 | **LinearSVC** | Supervised (SVM) | High-dimensional word space hyperplane separation | ~93% |
| 5 | **Soft Voting Ensemble** | Ensemble | Combines LR + RF + PAC + SVC with confidence weights | ~95% |
| 6 | **DistilBERT** | Deep Learning | Contextual language model — understands sentence meaning | ~96% |
| 7 | **K-Means Clustering** | Unsupervised | Topic cluster map — visual article landscape | N/A |
| 8 | **TF-IDF + Cosine Similarity** | Unsupervised | Similarity vs. credible reference texts | N/A |
| 9 | **Isolation Forest** | Unsupervised | Anomaly detection — flags unusual writing patterns | N/A |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit ≥ 1.32 |
| Backend | Python 3.10+ |
| ML Framework | scikit-learn, transformers, torch |
| NLP | NLTK, VADER, LIME |
| Data | WELFake (HuggingFace datasets), synthetic fallback |
| Visualization | Plotly (interactive), WordCloud, Matplotlib |
| Text Extraction | trafilatura, newspaper3k (URL), pytesseract (image OCR) |
| PDF Export | fpdf2 |
| **Claude AI** | **anthropic ≥ 0.25.0 — Credibility Brief + Ask Claude Q&A** |
| Deployment | Streamlit Cloud, Procfile |
| CI/CD | GitHub Actions |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip
- (Optional) Tesseract OCR for image text extraction

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/TruthLens.git
cd TruthLens/truthlens

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. (Optional) Download dataset and train models
python data/download_data.py        # Downloads WELFake from HuggingFace
python models/train_models.py       # Trains all ML models (~5-15 mins)

# 6. Launch TruthLens
streamlit run app.py
```

> **⚡ Quick Start:** Skip steps 5 and enable **Demo Mode** in the sidebar for instant results!

---

## 🤖 Claude AI Setup

TruthLens includes **Claude AI integration** for expert media-literacy analysis.

### Step 1 — Get an API Key
1. Sign up or log in at [console.anthropic.com](https://console.anthropic.com)
2. Navigate to **API Keys** → **Create Key**
3. Copy the key (starts with `sk-ant-...`)

### Step 2 — Configure TruthLens
```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your key
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

### Step 3 — Install Claude SDK
```bash
pip install anthropic python-dotenv
```

### Step 4 — Launch & Use
```bash
streamlit run app.py
```
After analyzing an article, scroll to the **🤖 Claude AI Analysis** section:
- **AI Credibility Brief** tab — auto-generated 3-bullet expert analysis
- **Ask Claude** tab — type any question or use quick-question chips

> **Note:** Claude features are gracefully disabled if no API key is set — the rest of the app works normally.

---

## 📁 Folder Structure

```
truthlens/
├── app.py                          ← Main Streamlit application
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
├── .gitignore
├── Procfile                        ← Streamlit Cloud deployment
│
├── .streamlit/
│   └── config.toml                 ← Dark theme settings
│
├── .github/
│   └── workflows/
│       └── deploy.yml              ← GitHub Actions CI/CD
│
├── models/
│   ├── __init__.py
│   ├── train_models.py             ← Train all ML models
│   ├── predict.py                  ← Unified prediction pipeline
│   └── trained/                    ← Saved .pkl model files (auto-generated)
│       ├── tfidf_vectorizer.pkl
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       ├── passive_aggressive.pkl
│       ├── linear_svc.pkl
│       ├── voting_ensemble.pkl
│       ├── kmeans.pkl
│       ├── pca.pkl
│       ├── isolation_forest.pkl
│       ├── cluster_data.json
│       └── training_metadata.json
│
├── data/
│   ├── download_data.py            ← Download WELFake / generate synthetic data
│   └── welfake_dataset.csv         ← Cached dataset (auto-generated)
│
├── utils/
│   ├── __init__.py
│   ├── text_preprocessor.py        ← NLTK cleaning + lemmatization
│   ├── feature_extractor.py        ← TF-IDF + linguistic + VADER features
│   └── credibility_scorer.py       ← TruthScore™ computation
│
└── assets/
    └── style.css                   ← Custom dark theme CSS
```

---

## 📸 Screenshots

> Screenshots will be added after first deployment.

| Feature | Preview |
|---------|---------|
| Home Dashboard | `[screenshot: welcome screen with feature cards]` |
| TruthScore Gauge | `[screenshot: animated gauge showing score 23 - FAKE]` |
| Algorithm Comparison | `[screenshot: bar chart with 5 model verdicts]` |
| LIME Explanation | `[screenshot: color-coded text with word weights]` |
| Cluster Explorer | `[screenshot: 2D scatter plot with new article marked]` |
| Student Module | `[screenshot: expandable learning cards]` |

---

## 🎮 Gamification System

| Badge | Points Required | Description |
|-------|----------------|-------------|
| 🥚 Novice | 0+ | Just getting started |
| 🔍 Fact-Finder | 50+ | Analyzed multiple articles |
| 🛡️ Truth Seeker | 150+ | Catching fake news consistently |
| 🏆 TruthLens Expert | 300+ | Master of misinformation detection |

**Earning Points:**
- +10 points per analysis
- +15 bonus points for catching high-confidence fake news (TruthScore < 30)

---

## 🌐 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, set main file: `truthlens/app.py`
4. Deploy!

### Local Docker
```bash
docker build -t truthlens .
docker run -p 8501:8501 truthlens
```

---

## ⚠️ Important Notes

- **No paid APIs required** — all analysis runs locally/offline
- **Demo Mode** provides instant results without GPU or trained models
- Train models once with `python models/train_models.py` for full accuracy
- pytesseract requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) to be installed separately for image analysis

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Sinega Selvakumar**
IBM Edunet AI Internship 2025

---

<div align="center">
<i>Built with ❤️ for students who want to see through the noise.</i>
</div>
