import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Initialize model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Predict sentiment
def predict_sentiment(text):
    # Preprocess input text
    text = preprocess(text)

    # Tokenize input
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # Model inference
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()

    # Apply softmax to get probabilities
    scores = softmax(scores)

    # Rank results
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    # Get label and score for the top sentiment
    top_sentiment = config.id2label[ranking[0]]
    top_score = np.round(float(scores[ranking[0]]), 4)

    return top_sentiment, top_score
