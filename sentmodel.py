import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'<br />', ' ', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.split()  # Split into words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]  # Remove stop words
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]  # Stemming
    return ' '.join(text)

# Apply the cleaning function to the reviews
df['cleaned_review'] = df['review'].apply(clean_text)

# Convert sentiment to numerical
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a new Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')

# Save the model and vectorizer
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
