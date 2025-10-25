from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)

# Free API URL for multiple cryptocurrencies
COIN_LIST = ['bitcoin','ethereum','binancecoin','ripple','cardano']
API_URL = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(COIN_LIST)}&vs_currencies=usd,inr"

# Historical simulation (for demo)
def generate_historical_prices(coin):
    np.random.seed(1)
    base_price = np.random.randint(1000, 50000)
    dates = pd.date_range(end=datetime.today(), periods=30)
    prices = base_price + np.random.randint(-1000, 1000, size=len(dates))
    return dates.strftime('%Y-%m-%d').tolist(), prices.tolist()

@app.route('/')
def index():
    response = requests.get(API_URL).json()
    crypto_prices = {coin.title(): response[coin] for coin in COIN_LIST}
    return render_template('index.html', crypto_prices=crypto_prices)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        crypto_name = request.form['crypto'].lower()
        currency_amount = float(request.form['amount'])
        
        # Simulated historical prices
        dates, prices = generate_historical_prices(crypto_name)
        
        # Train simple linear regression for prediction
        days = np.arange(len(prices)).reshape(-1,1)
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

# API endpoint for live update (AJAX)
@app.route('/live_prices')
def live_prices():
    response = requests.get(API_URL).json()
    crypto_prices = {coin.title(): response[coin] for coin in COIN_LIST}
    return jsonify(crypto_prices)

if __name__ == "__main__":
    app.run(debug=True)
