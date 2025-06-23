# 📌 Sentiment Analysis using TF-IDF and Logistic Regression (Single Block)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Sample Dataset (Replace with CSV if needed)
data = {
    "review": [
        "I loved the product! Great quality and amazing support.",
        "Terrible experience. Completely disappointed!",
        "Not bad, but could be better.",
        "Absolutely fantastic! I'm so happy.",
        "Worst purchase ever. Don't buy!",
        "Satisfied with the service.",
        "It broke after one day. Poor quality!",
        "Highly recommend to everyone!",
        "Waste of money. Very unhappy.",
        "The item is okay, works as expected."
    ],
    "sentiment": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Step 2: Preprocess Text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Step 3: Split Data
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Predictions & Evaluation
y_pred = model.predict(X_test_tfidf)

print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(
    y_test, y_pred, target_names=["Negative", "Positive"]
))

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
