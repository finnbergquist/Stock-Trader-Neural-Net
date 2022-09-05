import pandas as pd

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

# def main():
#     df = get_data('2016-12-25','2017-01-10',['AAPL','JPM','TSLA'], 'Adj Close', False)
#     print(df)
  
# if __name__=="__main__":
#     main()