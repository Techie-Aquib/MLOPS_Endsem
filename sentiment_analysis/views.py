from django.shortcuts import render
from .utils import predict_sentiment

def home(request):
    return render(request, 'home.html')

def analyze_sentiment(request):
    result = None
    sentiment = None
    score = None
    if request.method == 'POST':
        user_input = request.POST.get('text')
        sentiment, score = predict_sentiment(user_input)
        result = f"Sentiment: {sentiment} with confidence score: {score}"

    return render(request, 'result.html', {'result': result})
