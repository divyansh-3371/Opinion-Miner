import praw
import nltk
import re
import string
import emoji
import os
import time
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


reddit = praw.Reddit(
    client_id='IMEuyE0YQXJFikGp68rC1A',
    client_secret='ymfjcXOL1ZMLc3Z_MzkBLqXIimAMmw',
    user_agent='my_reddit_scraper_app by /u/Quirky_Writing9114'
)


def scrape_search_results(brand, sort_type, limit, start_timestamp, end_timestamp):
    print(f"\n🔍 Searching '{sort_type.upper()}' results for '{brand}'...")

    submissions = reddit.subreddit("all").search(query=brand, sort=sort_type, limit=10000)
    matched_posts = []

    for submission in submissions:
        created_utc = submission.created_utc
        if not (start_timestamp <= created_utc <= end_timestamp):
            continue

        title = submission.title
        body = submission.selftext or ""
        combined_text = f"{title} {body}".strip()

        if brand.lower() in combined_text.lower():
            matched_posts.append({
                "title": title,
                "body": body,
                "subreddit": submission.subreddit.display_name,
                "created_utc": datetime.utcfromtimestamp(created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "url": submission.url,
                "score": submission.score
            })

    
    sorted_posts = sorted(matched_posts, key=lambda x: x['score'], reverse=True)
    selected = sorted_posts[:limit]

    print(f"✅ Found {len(matched_posts)} posts in date range — using top {len(selected)} by score.")
    return selected


def save_all_preprocessed_text(results, brand):
    preprocessed_texts = set()

    for post in results:
        combined_text = f"{post['title']} {post['body']}".strip()
        cleaned = preprocess_text(combined_text)
        if cleaned and len(cleaned.split()) > 3: 
            preprocessed_texts.add(cleaned)

    os.makedirs("reddit_posts", exist_ok=True)
    filepath = os.path.join("reddit_posts", f"{brand}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        for line in sorted(preprocessed_texts):
            f.write(line + "\n")
    print(f"\n📝 Saved {len(preprocessed_texts)} unique, preprocessed posts to: {filepath}")
