from flask import Flask, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('logistic.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    cleaned_review = clean_text(review)
    review_vec = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vec)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
