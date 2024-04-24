import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Strategy():
    def __init__(self, ticker, start_date, end_date, market):
        self.start = start_date
        self.end = end_date
        self.ticker = ticker
        self.market = market
        self.data = yf.download(self.ticker, self.start, self.end)
        self.market_data = yf.download(self.market, self.start, self.end)

    def fetch_data(self):
        self.data = yf.download(self.ticker, self.start, self.end)

    def sma(self, short, long):
        '''
        Calculate short and long moving averages of stock price.
            short: short window
            long: long window
        '''

        self.data['SMA_Short'] = self.data['Close'].rolling(window=short).mean()
        self.data['SMA_Long'] = self.data['Close'].rolling(window=long).mean()

    def std_dev(self, window):
        '''
        Calculate the standard deviation of stock price (volatility) over a given window.
        '''

        self.data['Std_Dev'] = self.data['Close'].rolling(window=window).std()

    def z_score(self):
        '''
        Calculate the z-score of the stock price.
        '''

        self.data['Z_Score'] = (self.data['Close'] - self.data['SMA_Short']) / self.data['Std_Dev']

    def rsi(self):
        '''
        Calculate the RSI of a stock.
        '''

        delta = self.data['Close'].copy().diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down

        self.data['RSI'] = 100 - (100 / (1 + rs))

    def returns(self):
        '''
        Calculate the daily returns of a stock.
        '''
        self.data['Returns'] = self.data['Close'].pct_change()

    def drop_na(self):
        '''
        Drop NaNs.
        '''

        self.data.dropna(inplace=True)

    def target(self):
        '''
        Buy or sell target based on ff stock price goes up or down.
        '''

        self.data['Target'] = (self.data['Returns'].shift(-1) > 0).astype(int)


class MeanReversion(Strategy):

    def __init__(self, ticker, start_date, end_date, market):
        super().__init__(ticker, start_date, end_date, market)
        
        self.features = ['SMA_Short', 'SMA_Long', 'Std_Dev', 'Z_Score', 'RSI', 'Returns']
    
    def evaluate(self, model):
        '''
        Given a ML model, compare its returns vs. the market.
        '''
        self.sma(short=20, long=50)
        self.std_dev(window=20)
        self.z_score()
        self.rsi()
        self.returns()
        self.drop_na()
        self.target()

        X = self.data[self.features].copy()
        y = self.data['Target'].copy()

        # Split data into first 80% and last 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Calculate cumulative strategy returns on test data
        X_test['Predicted_Signal'] = y_pred
        X_test['Strategy_Returns'] = X_test['Returns'] * X_test['Predicted_Signal'].shift(1)
        cumulative_strategy_returns = (X_test['Strategy_Returns'] + 1).cumprod()

        # Calculate cumulative returns for the market
        market = self.market_data[self.market_data.index >= X_test.index[0]].copy()
        market['Returns'] = market['Close'].pct_change()
        cumulative_market_returns = (market['Returns'] + 1).cumprod()
        
        plt.figure(figsize=(10,5))
        plt.plot(cumulative_strategy_returns, label='Strategy Returns')
        plt.plot(cumulative_market_returns, label='Market Returns')
        plt.legend()
        plt.show()

        return X_test
        



    

    

    


