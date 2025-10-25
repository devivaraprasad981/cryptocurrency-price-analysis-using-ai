from flask import Flask, render_template, request, jsonify
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)

# Top 5 cryptocurrencies
COINS = ['bitcoin','ethereum','binancecoin','ripple','cardano']

# -----------------------------
# Get live prices from CoinGecko
# -----------------------------
def get_live_prices():
    try:
        ids = ','.join(COINS)
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr&ids={ids}"
        response = requests.get(url, timeout=10)
        data = response.json()
        crypto_prices = {}
        for coin in data:
            crypto_prices[coin['name']] = {
                'inr': coin['current_price'],
                'usd': round(coin['current_price']/80, 2)  # approximate USD
            }
        return crypto_prices
    except Exception as e:
        print("Error fetching live prices:", e)
        # fallback dummy values
        return {
            "Bitcoin": {"inr":0, "usd":0},
            "Ethereum": {"inr":0, "usd":0},
            "Binance Coin": {"inr":0, "usd":0},
            "Ripple": {"inr":0, "usd":0},
            "Cardano": {"inr":0, "usd":0}
        }

# -----------------------------
# Get historical prices (30 days)
# -----------------------------
def get_historical_prices(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=inr&days=30"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'prices' in data:
            prices = [p[1] for p in data['prices']]
            dates = [datetime.utcfromtimestamp(p[0]/1000).strftime('%d-%m') for p in data['prices']]
            return dates, prices
        else:
            return [f"Day {i+1}" for i in range(30)], [0 for _ in range(30)]
    except Exception as e:
        print("Error fetching historical prices:", e)
        return [f"Day {i+1}" for i in range(30)], [0 for _ in range(30)]

# -----------------------------
# Dashboard route
# -----------------------------
@app.route('/')
def index():
    crypto_prices = get_live_prices()
    dates, prices = get_historical_prices('bitcoin')  # show Bitcoin graph by default
    return render_template('index.html', crypto_prices=crypto_prices, dates=dates, prices=prices)

# -----------------------------
# Prediction route
# -----------------------------
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        crypto_name = request.form['crypto'].lower()
        amount_inr = float(request.form['amount'])

        coin_map = {
            'bitcoin':'bitcoin',
            'ethereum':'ethereum',
            'binancecoin':'binancecoin',
            'ripple':'ripple',
            'cardano':'cardano'
        }
        coin_id = coin_map.get(crypto_name, 'bitcoin')

        # Get historical prices
        dates, prices = get_historical_prices(coin_id)

        # Linear Regression to predict next day price
        days = [[i] for i in range(len(prices))]
        model = LinearRegression()
        model.fit(days, prices)
        predicted_price = model.predict([[len(prices)]])[0]

        crypto_amount = amount_inr / predicted_price if predicted_price != 0 else 0

        # Pass all variables to template
        return render_template('prediction.html',
                               crypto_name=crypto_name.title(),
                               currency_amount=amount_inr,
                               predicted_price=round(predicted_price,2),
                               crypto_amount=round(crypto_amount,6),
                               dates=dates,
                               prices=prices,
                               show_result=True)  # <- key fix
    return render_template('prediction.html', show_result=False)


# -----------------------------
# API endpoint for live JS update
# -----------------------------
@app.route('/live_prices')
def live_prices():
    return jsonify(get_live_prices())

if __name__ == "__main__":
    app.run(debug=True)
