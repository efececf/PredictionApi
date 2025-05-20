import requests
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from pymannkendall import original_test
from flask import Flask, jsonify
from flask_cors import CORS
from pmdarima import auto_arima

app = Flask(__name__)
CORS(app)

# Varsayılanlar
DEFAULT_YEARS = 1
DEFAULT_STEP_COUNT = 30

def filter_data_by_years(data, years=5):
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    return data[data.index >= cutoff_date]

def BringtheStockData(stock):
    try:
        response = requests.get(
            f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}&outputsize=full&apikey=6G6D8UJ8N7DG6NKI')
        response.raise_for_status()
        myStockData = response.json()

        if "Time Series (Daily)" not in myStockData:
            raise ValueError("API response does not contain data.")

        time_series = myStockData["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        df = df.dropna()
        return df['close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def reasonOfStationary(df):
    trend = original_test(df)
    fourier = fft(df.values)
    n = len(df)
    frequencies = np.fft.fftfreq(n)
    acf_max = pd.Series(df).autocorr(lag=7)

    if trend.trend != 'no trend':
        return 'trend'
    elif abs(acf_max) >= 0.5:
        return 'seasonality'
    return 'stationary'

@app.route('/predict/<stock>', methods=['GET'])
def predict_auto(stock):
    return _predict_core(stock, year=DEFAULT_YEARS, stepCount=DEFAULT_STEP_COUNT)

@app.route('/predict/<stock>/<int:year>/<int:stepCount>', methods=['GET'])
def predict_with_params(stock, year, stepCount):
    return _predict_core(stock, year=year, stepCount=stepCount)

def _predict_core(stock, year, stepCount):
    if not stock:
        return jsonify({"error": "Stock symbol is required."}), 400

    data = BringtheStockData(stock)
    if data is None or len(data) == 0:
        return jsonify({"error": "Could not fetch data for the given stock symbol."}), 400

    data = filter_data_by_years(data, years=year)
    if len(data) == 0:
        return jsonify({"error": f"No data in the last {year} year(s)."}), 400

    reason = reasonOfStationary(data)

    try:
        model = auto_arima(data, seasonal=(reason == 'seasonality'),m=7 if reason == 'seasonality' else 1, stepwise=False, suppress_warnings=True, error_action='ignore')
        forecast = model.predict(n_periods=stepCount)
    except Exception as e:
        return jsonify({"error": f"Model training failed: {str(e)}"}), 500

    return jsonify({
        "stationary": reason,
        "stockSymbol": stock,
        "predictedPrice": forecast.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
