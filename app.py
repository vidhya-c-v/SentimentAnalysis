from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import base64
import io
from urllib.error import HTTPError

# Download VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)
user_data = None

def scrape_finviz_data(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finviz_url + ticker

        try:
            req = Request(url=url, headers={'user-agent': 'my-app'})
            response = urlopen(req)
        except HTTPError as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')

        if news_table is not None:
            news_tables[ticker] = news_table
        else:
            print(f"No news table found for {ticker}")

    if not news_tables:
        print("No data fetched. Check ticker symbols and website structure.")
        return pd.DataFrame(columns=['ticker', 'date', 'time', 'title'])

    parsed_data = []

    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            title_cell = row.a
            if title_cell is not None:
                title = title_cell.text
                date_data = row.td.text.split(' ')

                if len(date_data) == 1:
                    time = date_data[0]
                else:
                    date = date_data[0]
                    time = date_data[1]

                parsed_data.append([ticker, date, time, title])

    return pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

def perform_sentiment_analysis(df):
    vader = SentimentIntensityAnalyzer()

    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['label'] = df['compound'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

    return df

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    accuracies = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        classification_rep = classification_report(y_test, predictions)

        accuracies[model_name] = accuracy

        print(f"{model_name} Accuracy: {accuracy}")
        print(f"{model_name} Classification Report:\n{classification_rep}\n")

    return accuracies

def generate_bar_graph(models, accuracies, title):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])

    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')

    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01,
                 f'{accuracy:.2%}', ha='center', color='black')

    plt.legend(['Random Forest', 'SVM', 'Logistic Regression'], loc='upper left')

    plt.show()

def plot_sentiment_distribution(data, ticker):
    filtered_data = data[data['ticker'] == ticker]

    if not filtered_data.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_data['compound'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)

        plt.title(f'Sentiment Distribution for Ticker: {ticker}')
        plt.xlabel('Compound Sentiment Score')
        plt.ylabel('Frequency')

        mean_sentiment = filtered_data['compound'].mean()
        plt.axvline(mean_sentiment, color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean Sentiment: {mean_sentiment:.2f}')

        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_str = base64.b64encode(img.getvalue()).decode()

        return img_str
    else:
        return None

def display_sentiment_message(sentiment):
    if sentiment == 'positive':
        return 'The sentiment is positive! <br>The ticker trend may move up. You may consider buying or investing in this stock.'
    elif sentiment == 'negative':
        return 'The sentiment is negative. <br>The ticker trend may move down. You may want to avoid investing in this stock or consider selling it.'
    elif sentiment == 'neutral':
        return 'The sentiment is neutral. <br>There might not be a strong movement in the ticker trend. Consider holding off on any investment decisions until more information is available.'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global user_data
    user_tickers = request.form['tickers'].upper().split(',')
    user_data = scrape_finviz_data(user_tickers)
    user_data = perform_sentiment_analysis(user_data)
    df = user_data.copy()
    
    #calculate mean sentiment scores
    
    mean_sentiments = user_data.groupby('ticker')['compound'].mean()

    # Plot bar chart of mean sentiment scores
    plt.figure(figsize=(10, 6))
    colors = ['skyblue' if x >= 0 else 'lightcoral' for x in mean_sentiments]
    plt.bar(mean_sentiments.index, mean_sentiments, color=colors)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Neutral Line')

    # Add labels and title
    plt.title('Mean Sentiment Scores for User-specified Tickers')
    plt.xlabel('Ticker')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.show()
    
    
    
    # Feature extraction
    
    X_user = user_data['compound'].values.reshape(-1, 1)
    label_encoder = LabelEncoder()
    y_user = label_encoder.fit_transform(user_data['label'])

    model_accuracies = train_and_evaluate_models(X_user, y_user)

    generate_bar_graph(['Random Forest', 'SVM', 'Logistic Regression'],
                       [model_accuracies['Random Forest'], model_accuracies['SVM'], model_accuracies['Logistic Regression']],
                       'Model Accuracy Comparison')

    return redirect(url_for('get_ticker'))

@app.route('/get_ticker')
def get_ticker():
    return render_template('result.html', sentiment_text="")

@app.route('/result', methods=['POST'])
def result():
    user_ticker = request.form['ticker']
    if user_data is not None:
        img_str = plot_sentiment_distribution(user_data, user_ticker)

        user_ticker_sentiment = user_data.loc[user_data['ticker'] == user_ticker, 'label'].iloc[0]
        sentiment_text = display_sentiment_message(user_ticker_sentiment)

        return render_template('histogram.html', img_str=img_str, sentiment_text=sentiment_text)
    else:
        return render_template('result.html', sentiment_text="No data available. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
