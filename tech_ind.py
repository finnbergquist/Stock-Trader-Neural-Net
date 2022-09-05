import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from readData import get_data

def SMA(start_date, end_date, prices, window_size):

    sma = prices.rolling(window=window_size,min_periods=window_size).mean()
    #sma.dropna(inplace=True)
    #print(sma)
    return sma

def EMA(start_date, end_date, prices, window_size):
    #E0 = SMA-n
    #a = 2/(n+1)
    #Et = ap(t) + (a - 1)Et-1

    ema = SMA(start_date, end_date, prices, window_size)
    a = 2/(window_size+1)
    
    for col, val in prices.iteritems():
        Ey = None
        day = 0
        for index, row in prices.iterrows():
            if (day < window_size): 
                day += 1
                continue
            if Ey == None: Ey = ema.iloc[day][col]
            E = (a * prices.loc[index, col]) + ((1 - a) * Ey)
            ema.loc[index, col] = E
            Ey = E
    return ema

def MACD(start_date, end_date, prices):
    ema12 = EMA(start_date, end_date, prices, 12)
    ema26 = EMA(start_date, end_date, prices, 26)
    macd = pd.DataFrame(ema12 - ema26)
    signal = pd.DataFrame(macd.ewm(span = 9, adjust = False).mean())
    hist = macd - signal
    macd.rename(columns = {list(prices)[0]:'MACD'}, inplace = True)
    signal.rename(columns = {list(prices)[0]:'SIG'}, inplace = True)
    hist.rename(columns = {list(prices)[0]:'HIST'}, inplace = True)
    frames =  [macd, signal, hist]
    #df = pd.concat(frames, join = 'inner', axis = 1)

    # ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    # ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)
    
    # price_cols = list(prices)
    # stock = price_cols[0]
    # ax1.plot(prices[stock])
    # ax2.plot(macd[stock], color = 'grey', linewidth = 1.5, label = 'MACD')
    # ax2.plot(signal[stock], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    # for index, row in prices.iterrows():
    #     if '-' in str(hist.loc[index, stock]):
    #         ax2.bar(index, hist.loc[index, stock], color = '#ef5350')
    #     else:
    #         ax2.bar(index, hist.loc[index, stock], color = '#26a69a')

    # plt.legend(loc = 'lower right')
    # plt.savefig('MACD')
    return frames

def BBP(start_date, end_date, prices, window_size):

    sma = SMA(start_date, end_date, prices, window_size)
    rolling_std = prices.rolling(window=window_size,min_periods=window_size).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (prices - bottom_band) / (top_band - bottom_band)
    bbp.rename(columns = {list(prices)[0]:'BBP'}, inplace = True)

    # figure, axis = plt.subplots(2, 1)
    # price_cols = list(prices)
    # stock = price_cols[0]
    # axis[0].plot(prices[stock])
    # axis[0].set_title("Normal Prices")
    # axis[1].plot(bbp[stock])
    # axis[1].set_title("BBP")
    # plt.savefig('BBP')
    return bbp

def RSI(prices, window_size):
    # Now we can calculate the RS and RSI all at once.
    rsi = prices.copy()
    # Pre-compute daily returns for repeated use.
    daily_rets = prices.copy()
    daily_rets.values[1:,:] = prices.values[1:,:] - prices.values[:-1,:]
    daily_rets.values[0,:] = np.nan
    # Pre-compute up and down returns.
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
    # Pre-compute up-day gains and down-day losses.
    up_gain = prices.copy()
    up_gain.loc[:,:] = 0
    up_gain.values[window_size:,:] = up_rets.values[window_size:,:] - up_rets.values[:-window_size,:]
    down_loss = prices.copy()
    down_loss.loc[:,:] = 0
    down_loss.values[window_size:,:] = down_rets.values[window_size:,:] - down_rets.values[:-window_size,:]

    rs = (up_gain / window_size) / (down_loss / window_size)
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:window_size,:] = np.nan

    # Inf results mean down_loss was 0.  Those should be RSI 100.
    rsi[rsi == np.inf] = 100
    rsi.rename(columns = {list(prices)[0]:'RSI'}, inplace = True)
    #print(rsi)
    # figure, axis = plt.subplots(2, 1)
    # price_cols = list(prices)
    # stock = price_cols[0]
    # axis[0].plot(prices[stock])
    # axis[0].set_title("Normal Prices")
    # axis[1].plot(rsi[stock])
    # axis[1].set_title("RSI")
    # plt.savefig('RSI')
    return rsi

def x_day_low(start_date, end_date, prices, window_size):
    lows = prices.rolling(window_size, min_periods=1).min()
    # price_cols = list(prices)
    # stock = price_cols[0]
    # figure, axis = plt.subplots(2, 1)
    # axis[0].plot(prices[stock])
    # axis[0].set_title("Normal Prices")
    # axis[1].plot(lows[stock])
    # axis[1].set_title("5 Day Lows")
    # plt.savefig('X Day Low')
    return lows


def main():
    symbols = ['DIS','JPM','TSLA']
    start_date = '2018-01-01'
    end_date = '2019-12-31'
    window_size = 14
    prices = get_data(start_date, end_date, symbols)
    #macd = MACD(start_date, end_date, prices, window_size)
    bbands = BBands(start_date, end_date, prices, window_size)
    print(bbands.to_string())
  
if __name__=="__main__":
    main()