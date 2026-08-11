# 🧠 Opinion Miner

### AI-Powered Reddit Sentiment Analysis & Opinion Mining Platform

**Opinion Miner** is an NLP-powered sentiment analysis platform that extracts and analyzes public opinions from Reddit.

The system combines **Reddit scraping, natural language preprocessing, transformer-based sentiment classification, and interactive Streamlit dashboards** to identify how people feel about a company, product, service, or topic.

Instead of manually reading hundreds of Reddit discussions, Opinion Miner automatically collects posts, processes the text, classifies sentiment, and presents the results through an interactive dashboard.

---

## 🚀 Features

### 🔎 Reddit Opinion Mining

Opinion Miner collects Reddit posts based on a user-defined search query.

Users can analyze:

* Companies
* Brands
* Products
* Services
* Topics
* Public discussions

The scraper supports collecting Reddit content for sentiment analysis and further NLP processing.

---

### 📊 Sentiment Analysis

The project uses a transformer-based NLP pipeline to classify Reddit content according to sentiment.

The workflow processes each post through:

```text
Reddit Post
     ↓
Text Cleaning
     ↓
Emoji Processing
     ↓
Normalization
     ↓
Tokenization
     ↓
Transformer Model
     ↓
Sentiment Classification
```

The resulting sentiment information can then be aggregated to understand the overall public opinion.

---

### 🤖 Transformer-Based NLP

Opinion Miner uses modern NLP techniques rather than relying only on traditional keyword-based sentiment analysis.

The training pipeline includes:

* Transformer-based language models
* Tokenization
* Sequence classification
* Preprocessed datasets
* Model training
* Validation
* Inference

This allows the system to capture contextual meaning in user-generated text.

---

### 🧹 Advanced Text Preprocessing

Reddit content often contains:

* Slang
* Emojis
* URLs
* Special characters
* Repeated characters
* Informal language
* User mentions

The preprocessing pipeline prepares this noisy social-media text before it reaches the sentiment model.

Processing includes:

* Text normalization
* Emoji handling
* Stopword processing
* Lemmatization
* Special-character removal
* Tokenization

---

### 📈 Sentiment Insights

The platform can transform individual predictions into higher-level insights.

For example:

```text
                 Reddit Discussions
                         │
                         ▼
                Sentiment Analysis
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Positive     Neutral     Negative
             │           │           │
             └───────────┼───────────┘
                         ▼
                  Overall Opinion
                         │
                         ▼
                  User Insights
```

This makes the system useful for understanding public perception rather than simply labeling individual sentences.

---

### 🔐 Streamlit Authentication

The application includes a Streamlit-based login system.

The repository includes:

```text
login.py
```

which provides the entry point for the authenticated application experience.

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Streamlit Web App  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                 ┌───────────────┐     ┌───────────────┐
                 │ Search Query  │     │ Authentication │
                 └───────┬───────┘     └───────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Reddit Scraper│
                 │     PRAW      │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Raw Reddit    │
                 │     Data      │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Preprocessing │
                 │     NLP       │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Transformer   │
                 │ Sentiment     │
                 │    Model      │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Sentiment     │
                 │ Predictions   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Visualization │
                 │ & Insights    │
                 └───────────────┘
```

---

## 🔬 Machine Learning Pipeline

The ML pipeline consists of four major stages.

### 1. Data Collection

Reddit posts are collected using the Reddit API.

```text
Search Query
     ↓
Reddit API
     ↓
Posts & Comments
     ↓
Raw Dataset
```

---

### 2. Data Preprocessing

Raw social-media text is cleaned and transformed into a format suitable for model training.

```text
Raw Text
   ↓
Cleaning
   ↓
Emoji Processing
   ↓
Normalization
   ↓
Stopword Processing
   ↓
Lemmatization
   ↓
Tokenization
```

---

### 3. Model Training

The processed dataset is used to train a transformer-based sentiment classification model.

```text
Preprocessed Dataset
        ↓
Tokenizer
        ↓
Transformer
        ↓
Classification Head
        ↓
Training
        ↓
Validation
        ↓
Trained Model
```

---

### 4. Inference

After training, new Reddit posts can be passed through the trained model.

```text
New Reddit Post
      ↓
Preprocessing
      ↓
Tokenizer
      ↓
Trained Transformer
      ↓
Prediction
      ↓
Sentiment Label
```

---

## 📚 Dataset

The repository contains multiple datasets and preprocessing artifacts used throughout the project.

Relevant files include:

```text
combined_sentiment_data.csv
twitter_training.csv
preprocessed_data.arrow
```

The datasets can be used for training and evaluating the sentiment classification pipeline.

The project also uses Reddit data collected through the scraping pipeline.

---

## 🛠️ Tech Stack

### Programming

* **Python**

### Machine Learning & NLP

* **PyTorch**
* **Hugging Face Transformers**
* **Hugging Face Datasets**
* **NLTK**

### Data Processing

* **Pandas**
* **NumPy**
* Apache Arrow datasets

### Data Collection

* **PRAW**
* Reddit API

### Application

* **Streamlit**

### Visualization

* Streamlit visualizations
* NLP sentiment analysis outputs

### Authentication

* Streamlit-based login system
* SQLite database

---

## 📂 Project Structure

```text
Opinion-Miner/
│
├── output/
│
├── reddit_posts/
│
├── combined_sentiment_data.csv
├── twitter_training.csv
├── preprocessed_data.arrow
│
├── inference.py
├── train.py
├── val.py
│
├── reddit_scrap.py
├── scraper.py
│
├── main.py
├── login.py
│
├── requirements.txt
│
├── loading.json
├── welcome.json
│
├── users.db
│
├── test.txt
└── README.md
```

The repository currently contains these major components, including separate training, validation, inference, scraping, and application files.

---

## 📌 File Overview

| File / Directory              | Purpose                            |
| ----------------------------- | ---------------------------------- |
| `main.py`                     | Main application workflow          |
| `login.py`                    | Streamlit authentication interface |
| `reddit_scrap.py`             | Reddit data collection             |
| `scraper.py`                  | Scraping-related functionality     |
| `train.py`                    | Sentiment model training           |
| `val.py`                      | Model validation                   |
| `inference.py`                | Model inference/prediction         |
| `combined_sentiment_data.csv` | Combined sentiment dataset         |
| `twitter_training.csv`        | Training data                      |
| `preprocessed_data.arrow`     | Preprocessed dataset               |
| `reddit_posts/`               | Reddit data                        |
| `output/`                     | Generated outputs                  |
| `users.db`                    | Application user database          |
| `requirements.txt`            | Python dependencies                |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/divyansh-3371/Opinion-Miner.git
cd Opinion-Miner
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv myenv
myenv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration

Opinion Miner requires access to the Reddit API for collecting Reddit data.

Create your Reddit API credentials and configure them through environment variables or your application's configuration.

Example:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

**Never commit API credentials or secrets to GitHub.**

---

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run login.py
```

The application will open in your browser.

After authentication, users can access the sentiment-analysis workflow.

---

## 🧪 Training the Model

To train the sentiment classification model:

```bash
python train.py
```

The training workflow uses the prepared sentiment datasets and transformer-based NLP architecture.

---

## 🔍 Model Validation

Model validation can be performed using:

```bash
python val.py
```

This allows the trained model to be evaluated on validation data.

---

## 🤖 Running Inference

For sentiment predictions on new data:

```bash
python inference.py
```

The inference pipeline processes input text and generates sentiment predictions using the trained model.

---

## 📊 End-to-End Workflow

```text
                         USER
                           │
                           ▼
                  Enter Company / Topic
                           │
                           ▼
                    Reddit Search
                           │
                           ▼
                  Reddit Data Collection
                           │
                           ▼
                    Text Cleaning
                           │
                           ▼
                   NLP Preprocessing
                           │
                           ▼
                 Transformer Inference
                           │
                           ▼
                 Sentiment Classification
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Positive       Neutral       Negative
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  Sentiment Aggregation
                           │
                           ▼
                  Interactive Dashboard
                           │
                           ▼
                     User Insights
```

---

## 💡 Use Cases

Opinion Miner can be used for:

* 📈 Market sentiment analysis
* 🏢 Brand perception analysis
* 🛍️ Product feedback analysis
* 📢 Customer opinion monitoring
* 🔎 Social listening
* 📊 Public opinion research
* 💬 Reddit community analysis
* 🧠 NLP research
* 📉 Sentiment-based trend analysis

---

## 🎯 Why Opinion Miner?

Traditional sentiment analysis often works on static datasets.

Opinion Miner adds a **data collection + machine learning + application layer**:

```text
Traditional NLP
      │
      ▼
Static Dataset → Sentiment Model → Prediction
```

Opinion Miner:

```text
Live/Collected Reddit Data
          │
          ▼
     Data Scraping
          │
          ▼
   NLP Preprocessing
          │
          ▼
 Transformer Model
          │
          ▼
 Sentiment Analysis
          │
          ▼
 Interactive Application
```

This makes the project closer to a complete **end-to-end NLP application** rather than only a machine-learning model.

---

## 🔮 Future Improvements

Potential improvements include:

* [ ] Real-time Reddit sentiment monitoring
* [ ] Live sentiment dashboards
* [ ] Sentiment trend analysis over time
* [ ] Topic modeling
* [ ] Aspect-based sentiment analysis
* [ ] Emotion classification
* [ ] Multilingual sentiment analysis
* [ ] Reddit comment-level analysis
* [ ] News sentiment integration
* [ ] Twitter/X sentiment integration
* [ ] Stock-price vs. sentiment correlation
* [ ] LLM-generated sentiment summaries
* [ ] Automated market sentiment reports
* [ ] Cloud deployment
* [ ] Redis-based caching
* [ ] Model monitoring and evaluation

---

## 🔐 Security Considerations

The following files and information should **not** be publicly committed if they contain sensitive information:

```text
.env
API credentials
Reddit client secrets
Authentication secrets
Private user information
```

For production deployments, credentials should be managed through environment variables or a secure secrets manager.

---

## ⚠️ Disclaimer

Opinion Miner is intended for **educational, research, and analytical purposes**.

Sentiment predictions represent opinions expressed in the analyzed data and should not be treated as definitive statements about a company, product, market, or investment.

---

## 👨‍💻 Author

**Divyansh Bansal**

GitHub: [@divyansh-3371](https://github.com/divyansh-3371)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐.

**Repository:** [Opinion-Miner](https://github.com/divyansh-3371/Opinion-Miner)
