from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)

# List of top 5 cryptocurrencies
COINS = ['bitcoin','ethereum','binancecoin','ripple','cardano']

# CoinGecko API for live prices
API_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr&ids=" + ','.join(COINS)

# Function to get live prices
def get_live_prices():
    try:
        response = requests.get(API_URL).json()  # parse JSON
        crypto_prices = {}
        for coin in response:
            # Ensure coin is a dict
            if isinstance(coin, dict):
                crypto_prices[coin['name']] = {
                    'inr': coin['current_price'],
                    'usd': round(coin['current_price']/80,2)
                }
        return crypto_prices
    except Exception as e:
        print("Error fetching live prices:", e)
        # return default/fallback values
        return {
            "Bitcoin": {"inr":0, "usd":0},
            "Ethereum": {"inr":0, "usd":0},
            "Binance Coin": {"inr":0, "usd":0},
            "Ripple": {"inr":0, "usd":0},
            "Cardano": {"inr":0, "usd":0}
        }


# Function to get historical prices for graph (last 30 days)
def get_historical_prices(coin_id):
    HIST_URL = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=inr&days=30"
    response = requests.get(HIST_URL).json()
    prices = [p[1] for p in response['prices']]
    dates = [datetime.utcfromtimestamp(p[0]/1000).strftime('%d-%m') for p in response['prices']]
    return dates, prices

@app.route('/')
def index():
    crypto_prices = get_live_prices()
    # For example, show Bitcoin historical prices in dashboard
    dates, prices = get_historical_prices('bitcoin')
    return render_template('index.html', crypto_prices=crypto_prices, dates=dates, prices=prices)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        crypto_name = request.form['crypto'].lower()
        currency_amount = float(request.form['amount'])
        # Historical prices for prediction
        dates, prices = get_historical_prices(crypto_name)
        days = [[i] for i in range(len(prices))]
        model = LinearRegression()
        model.fit(days, prices)
        predicted_price = model.predict([[len(prices)]])[0]
        crypto_amount = currency_amount / predicted_price
        return render_template('prediction.html', crypto_name=crypto_name.title(),
                               currency_amount=currency_amount,
                               predicted_price=round(predicted_price,2),
                               crypto_amount=round(crypto_amount,6),
                               dates=dates, prices=prices)
    return render_template('prediction.html')

@app.route('/live_prices')
def live_prices():
    return jsonify(get_live_prices())

if __name__ == "__main__":
    app.run(debug=True)
