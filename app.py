from flask import Flask, request, jsonify, render_template,send_file
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from rake_nltk import Rake
import json

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    lang = request.json.get('lang', 'en')  
    
    
    if lang != 'en':
        text = translator.translate(text, src=lang, dest='en').text
    
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment_text = "This comment is positive" if polarity >= 0 else "This comment is negative"
    
    
    vader_score = analyzer.polarity_scores(text)
    compound_score = vader_score['compound']
    
    
    if compound_score >= VADER_POS_THRESHOLD:
        vader_sentiment = "positive"
    elif compound_score <= VADER_NEG_THRESHOLD:
        vader_sentiment = "negative"
    else:
        vader_sentiment = "neutral"
        
    negative_words = [word for word in text.split() if analyzer.polarity_scores(word)['compound'] < 0]
    
    
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
 
    return jsonify({
        'polarity': polarity,
        'sentiment_text': sentiment_text,
        'vader_sentiment': vader_sentiment,
        'vader': vader_score,
        'negativeWords': negative_words,
        'keywords': keywords
    })

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    texts = request.json['texts']  
    results = []
    for text in texts:
        
        lang = request.json.get('lang', 'en')
        if lang != 'en':
            text = translator.translate(text, src=lang, dest='en').text
        
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment_text = "Positive" if polarity >= 0 else "Negative"
        
        
        vader_score = analyzer.polarity_scores(text)
        
        
        if vader_score['compound'] >= VADER_POS_THRESHOLD:
            vader_sentiment = "positive"
        elif vader_score['compound'] <= VADER_NEG_THRESHOLD:
            vader_sentiment = "negative"
        else:
            vader_sentiment = "neutral"
        
        results.append({
            'text': text,
            'polarity': polarity,
            'sentiment_text': sentiment_text,
            'vader_sentiment': vader_sentiment,
            'vader': vader_score
        })
    
    return jsonify(results)

@app.route('/chart-data', methods=['POST'])
def chart_data():
    texts = request.json['texts']
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    
    for text in texts:
        lang = request.json.get('lang', 'en')
        if lang != 'en':
            text = translator.translate(text, src=lang, dest='en').text
        
        vader_score = analyzer.polarity_scores(text)
        compound_score = vader_score['compound']
        
        if compound_score >= VADER_POS_THRESHOLD:
            sentiment_counts["positive"] += 1
        elif compound_score <= VADER_NEG_THRESHOLD:
            sentiment_counts["negative"] += 1
        else:
            sentiment_counts["neutral"] += 1
    
    return jsonify(sentiment_counts)

if __name__ == '__main__':
    app.run(debug=True)
