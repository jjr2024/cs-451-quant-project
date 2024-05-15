import yfinance as yf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(tickers, start_date=None, end_date=None, period=None, test_size=0.1):
    '''
    Combines data of all tickers into a single dataframe for X_train. X_test is a list of dataframes for each ticker.
    '''
    #features = ['SMA_20', 'SMA_50', 'Std_Dev', 'Z_Score', 'RSI', 'Close', 'TTM_P/E']
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    X_scalers = []
    y_scalers = []
    for t in tickers:
        ticker_data = yf.Ticker(t)
        data = ticker_data.history(start=start_date, end=end_date, period=period)

        # Calculate moving averages and std
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_100'] = data['Close'].rolling(window=100).mean()
        data['SMA_250'] = data['Close'].rolling(window=250).mean()
        data['Std_Dev'] = data['Close'].rolling(window=20).std()

        # Calculate the z-score
        data['Z_Score'] = (data['Close'] - data['SMA_20']) / data['Std_Dev']

        # Calculate RSI
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down

        data['RSI'] = 100 - (100 / (1 + rs))
        # Calculate TTM EPS and P/E
        eps = ticker_data.get_earnings_dates(limit=60)
        eps = eps.loc[~eps.index.duplicated(keep='first'), :]
        eps = eps[(eps.index >= (data.index[0]-relativedelta(years=1))) & (eps.index <= data.index[-1])]
        if t == 'DUK':
            eps.loc[eps.index == pd.to_datetime('2024-05-07').date(), 'Reported EPS'] = 1.44
        eps = eps.iloc[::-1]
        eps['TTM'] = eps['Reported EPS'].rolling(window=4).sum()
        eps.index = eps.index.date
        idx = pd.date_range(eps.index[0], eps.index[-1])
        eps = eps.reindex(idx.date, fill_value=np.nan)
        data.index = data.index.date
        data['TTM_EPS'] = eps['TTM'].copy()
        data[data['TTM_EPS'].notna()]
        data['TTM_EPS'] = data['TTM_EPS'].ffill()
        data['TTM_EPS'] = data['TTM_EPS'].fillna(eps['TTM'].loc[eps['TTM'].notna()].iloc[0])
        data['TTM_P/E'] = data['Close'] / data['TTM_EPS']

        # Calculate the daily returns
        data['Returns'] = data['Close'].pct_change()

        # Drop any NaNs

        # If stock price goes up or down
        data['Target'] = data['Close'].shift(-1)
        data.dropna(inplace=True)
        #features = ['Ticker', 'SMA_20', 'SMA_50', 'Std_Dev', 'Z_Score', 'RSI', 'Returns']
        X = data.loc[:, data.columns != 'Target']
        y = data.iloc[:, (data.shape[1]-1):(data.shape[1])]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
        ss1 = StandardScaler()
        ss2 = StandardScaler()
        #mm = MinMaxScaler()

        X_train_ss = pd.DataFrame(ss1.fit_transform(X_train), index=X_train.index, columns=X_train.columns) # fit ss and transform X_train
        y_train_mm = pd.DataFrame(ss2.fit_transform(y_train), index=y_train.index, columns=y_train.columns) # fit mm and transform y_train
        X_test_ss = pd.DataFrame(ss1.transform(X_test), index=X_test.index, columns=X_test.columns) # transform X_test with fitted ss
        y_test_mm = pd.DataFrame(ss2.transform(y_test), index=y_test.index, columns=y_test.columns) # transform y_test with fitted mm
        X_train_ss['Ticker'] = t
        X_test_ss['Ticker'] = t
        print(X_train.shape)
        X_train_list.append(X_train_ss)
        y_train_list.append(y_train_mm)
        X_test_list.append(X_test_ss)
        y_test_list.append(y_test_mm)
        X_scalers.append(ss1)
        y_scalers.append(ss2)
    batch_size = X_train_list[0].shape[0]

    return pd.concat(X_train_list, ignore_index=False), pd.concat(y_train_list, ignore_index=False), X_test_list, y_test_list, X_scalers, y_scalers, batch_size