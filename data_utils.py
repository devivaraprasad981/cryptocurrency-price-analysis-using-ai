import requests
import pandas as pd
import time


COINGECKO_BASE = 'https://api.coingecko.com/api/v3'


def fetch_historical_prices(coin_id='bitcoin', vs_currency='usd', days=30):
"""Fetch historical market data (prices) from CoinGecko.
Returns DataFrame with columns: timestamp (ms), price (float)
"""
endpoint = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
params = {'vs_currency': vs_currency, 'days': days}
r = requests.get(endpoint, params=params)
r.raise_for_status()
data = r.json()
prices = data.get('prices', []) # list of [timestamp, price]
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
# convert timestamp to datetime if needed
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
return df




def make_features(df):
"""Create lag features and moving averages for ML model."""
df_sorted = df.sort_values('timestamp').reset_index(drop=True)
df_feat = df_sorted.copy()
df_feat['price_lag1'] = df_feat['price'].shift(1)
df_feat['price_lag2'] = df_feat['price'].shift(2)
df_feat['ma3'] = df_feat['price'].rolling(window=3).mean()
df_feat['ma7'] = df_feat['price'].rolling(window=7).mean()
df_feat = df_feat.dropna().reset_index(drop=True)
X = df_feat[['price_lag1', 'price_lag2', 'ma3', 'ma7']]
y = df_feat['price']
return X, y