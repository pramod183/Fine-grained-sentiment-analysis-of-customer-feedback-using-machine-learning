# 💬 Fine-Grained Sentiment Analysis of Customer Feedback Using Machine Learning

> A Django web application that performs **5-class fine-grained sentiment analysis** on customer feedback and tweets using three deep learning models — CNN, ANN, and LSTM — with GloVe word embeddings, allowing side-by-side model comparison.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![GloVe](https://img.shields.io/badge/GloVe-Embeddings-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📋 Overview

Unlike standard binary (positive/negative) sentiment analysis, this project performs **fine-grained 5-class sentiment classification** — distinguishing between frustrated, negative, neutral, positive, and satisfied feedback. Three deep learning architectures are trained and evaluated side by side, giving a comprehensive comparison of model performance.

**Key highlights:**
- ✅ **5-class sentiment classification** — frustrated / negative / neutral / positive / satisfied
- ✅ **3 deep learning models** trained and compared: CNN, ANN, LSTM
- ✅ **GloVe 100-dimensional word embeddings** for rich semantic text representation
- ✅ **Django web app** with separate Admin and User portals
- ✅ **Twitter/Tweet search integration** — search keywords and classify live tweets
- ✅ **Performance comparison charts** — accuracy, precision, recall, F1-score bar graphs per model
- ✅ Pre-trained models saved as `.h5` files for instant inference

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Django Web App                        │
│  ┌─────────────────┐        ┌──────────────────────┐   │
│  │   Admin Portal  │        │    User Portal       │   │
│  │  - Train Models │        │  - Search Tweets     │   │
│  │  - View Accuracy│        │  - View Predictions  │   │
│  │  - Compare ML   │        │  - Sentiment Results │   │
│  └────────┬────────┘        └──────────┬───────────┘   │
└───────────┼──────────────────────────────┼──────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────┐    ┌─────────────────────────┐
│   Model Training      │    │   Inference (LSTM)      │
│  Train_CNN.py  → CNN  │    │   LSTM.py               │
│  Train_ANN.py  → ANN  │    │   get_predictions(text) │
│  Train_LSTM.py → LSTM │    │   → 5-class label       │
└───────────────────────┘    └─────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────┐
│              GloVe Embeddings (100-dim)               │
│   Word → 100-dimensional semantic vector              │
│   Pre-trained on 6B tokens from Wikipedia + Gigaword  │
└───────────────────────────────────────────────────────┘
```

---

## 🤖 Models

### 1. CNN (Convolutional Neural Network)
```
Embedding (GloVe 100d, non-trainable)
→ Conv1D (128 filters, kernel=3, ReLU)
→ GlobalMaxPooling1D
→ Dense (64, ReLU)
→ Dense (5, Softmax)
```
- Sequence length: 100 tokens
- Embedding: GloVe 100-dim (frozen weights)
- Epochs: 20 | Batch size: 128
- Optimizer: Adam | Loss: Categorical Crossentropy

### 2. ANN (Artificial Neural Network)
```
Flatten (input_shape=500)
→ Dense (128, ReLU)
→ Dense (5, Softmax)
```
- Sequence length: 500 tokens
- Epochs: 2 | Batch size: 32
- Optimizer: Adam | Loss: Categorical Crossentropy

### 3. LSTM (Long Short-Term Memory)
```
Embedding (GloVe 100d)
→ LSTM layers
→ Dense (5, Softmax)
```
- Sequence length: 100 tokens
- Embedding size: 100-dim GloVe
- Epochs: 20 | Batch size: 64
- Saved as: `lstm_model.h5` (used for live inference)

---

## 🏷️ Sentiment Classes

| Label | Meaning |
|---|---|
| 😤 **frustrated** | Strong dissatisfaction, anger |
| 😞 **negative** | Unhappy, complaints |
| 😐 **neutral** | Neither positive nor negative |
| 😊 **positive** | Happy, pleased |
| 😄 **satisfied** | Highly content, delighted |

---

## 📊 Model Evaluation

All three models are evaluated on the same 25% test split using:

| Metric | Description |
|---|---|
| Accuracy | % of correctly classified samples |
| Precision | Ability to avoid false positives (macro average) |
| Recall | Ability to find all positive cases (macro average) |
| F1 Score | Harmonic mean of Precision and Recall (macro average) |

Results are stored in the database and visualised as **bar charts comparing all 3 models** across all 4 metrics.

---

## 🌐 Web App Features

### Admin Panel
- Login with `admin` / `admin`
- Train CNN, ANN, or LSTM model with one click
- View accuracy comparison table
- View performance bar charts (Accuracy / Precision / Recall / F1)

### User Panel
- Register / Login
- Search tweets by keyword → real-time sentiment classification
- View classified tweets with sentiment labels
- View sentiment frequency distribution graph

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Django |
| Deep Learning | TensorFlow / Keras |
| Text Processing | Keras Tokenizer, Pad Sequences |
| Word Embeddings | GloVe 6B 100-dimensional vectors |
| Models | CNN, ANN, LSTM |
| Database | Django ORM (SQLite default) |
| Visualisation | Matplotlib |
| Tweet Search | Tweepy (Twitter API) |
| Language | Python 3.8+ |

---

## 📁 Project Structure

```
sentiment-analysis/
├── views.py              # Django views — routing, model training triggers
├── models.py             # Database models (user, tweets, accuracysc)
├── urls.py               # URL routing
├── Train_CNN.py          # CNN training + evaluation pipeline
├── Train_ANN.py          # ANN training + evaluation pipeline
├── Train_LSTM.py         # LSTM training pipeline
├── Train_LSTM2.py        # LSTM variant
├── LSTM.py               # Inference — get_predictions() for live classification
├── TweetSearch.py        # Tweepy Twitter search integration
├── Freq.py               # Sentiment frequency counter
├── Graphs.py             # Graph generation utilities
├── bargraph.py           # Bar chart rendering helper
├── train.csv             # Labelled training dataset
├── tokenizer.pickle      # Saved tokenizer (after training)
├── cnn_model.h5          # Saved CNN model weights
├── lstm_model.h5         # Saved LSTM model weights
├── settings.py           # Django settings
├── manage.py             # Django management commands
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install django tensorflow keras numpy pandas matplotlib scikit-learn tweepy pillow tqdm
```

### GloVe Embeddings
Download GloVe vectors (required for CNN and LSTM training):
```bash
# Download from: https://nlp.stanford.edu/projects/glove/
# Get: glove.6B.zip → extract glove.6B.100d.txt
# Place in: data/glove.6B.100d.txt
```

### Run the App
```bash
git clone https://github.com/pramod183/fine-grained-sentiment-analysis.git
cd fine-grained-sentiment-analysis
python manage.py migrate
python manage.py runserver
```
Open `http://127.0.0.1:8000` in your browser.

### Training the Models
1. Log in as **Admin** (username: `admin`, password: `admin`)
2. Go to **Training Page**
3. Click **Train CNN** / **Train ANN** / **Train LSTM**
4. Once complete, go to **View Accuracy** to compare results

---

## 🔑 Key Learnings

- **Fine-grained vs binary sentiment**: 5-class classification is significantly harder than binary — model accuracy naturally lower; F1-score is the most meaningful metric due to class imbalance
- **GloVe embeddings outperform random initialisation** — pre-trained semantic vectors give the model a head start, especially with limited training data
- **CNN vs LSTM for text**: CNN with GlobalMaxPooling captures local n-gram features efficiently; LSTM captures sequential dependencies — both have strengths depending on text length
- **Freezing embedding weights** (`trainable=False`) prevents the pre-trained GloVe vectors from being corrupted during early training epochs
- **Tokenizer persistence** — saving the tokenizer as `tokenizer.pickle` is essential; the same tokenizer used during training must be used during inference or predictions will be meaningless

---

## 👤 Author

**Pramod Baddhipadugu**
Bachelors in Computer Science · Lovely professioanl uniersity
[pramodreddy181@gmail.com](mailto:pramodreddy181@gmail.com)
