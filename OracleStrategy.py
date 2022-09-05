import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import assess_portfolio

def get_data(start_date, end_date, symbols, column_name = 'Adj Close', include_spy=False):

    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if include_spy:
        df_new = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True, na_values=['nan'], usecols=['Date', column_name])
        df = df.join(df_new, how='inner')
        df = df.rename(columns={'Adj Close' : 'SPY'})
    else:
        df_new = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True, na_values=['nan'], usecols=['Date'])
        df = df.join(df_new, how='inner')

    for stock in symbols:

        df_new = pd.read_csv("data/" + stock + ".csv", index_col="Date", parse_dates=True, na_values=['nan'], usecols=['Date', column_name])
        df = df.join(df_new, how='left') 
        df = df.rename(columns={'Adj Close' : stock})

    return df

class OracleStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(self, start_date = '2018-01-01', end_date = '2019-12-31', symbol = 'DIS', starting_cash = 200000):
        # Inputs represent the date range to consider, the single stock to trade, and the starting portfolio value.
        #
        # Return a date-indexed DataFrame with a single column containing the desired trade for that date.
        # Given the position limits, the only possible values are -2000, -1000, 0, 1000, 2000.

        prices = get_data(start_date, end_date, [symbol])
        daily_rets = prices.copy()
        abs_rets = prices.copy()
        abs_rets.values[1:,:] = abs(prices.values[1:,:] - prices.values[:-1,:])
        daily_rets.values[1:,:] = prices.values[1:,:] - prices.values[:-1,:]
        daily_rets.values[0,:] = np.nan
        abs_rets.values[0,:] = np.nan
        Ovals = abs_rets.cumsum()
        bline = daily_rets.cumsum()
        bline.iloc[0] = 0

        # print("Orcale Profit: " + str(Ovals[symbol].iloc[-1] * 1000))
        # print("Baseline Profit: " + str(bline[symbol].iloc[-1] * 1000))
        # print("\n")

        # assess_portfolio((Ovals[symbol] * 1000) + starting_cash)
        # print("\n")
        assess_portfolio((bline[symbol] * 1000) + starting_cash)
        #plot
        # plt.plot(Ovals[symbol] * 1000, 'r-', label='Oracle Trader')
        # plt.plot(bline[symbol] * 1000, 'b-', label='Buy and Hold Baseline')
        # plt.xlabel('Time')
        # plt.ylabel('Portfolio Value')
        # plt.savefig('Oracle_vs_baseline')

        #[symbol].iloc[-1]
        return (bline * 1000) + starting_cash

def main():
    o = OracleStrategy()
    print(o.test())

  
if __name__=="__main__":
    main()