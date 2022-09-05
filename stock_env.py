import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn.metrics import log_loss
from readData import get_data
from tech_ind import MACD, RSI, BBP
from backtest import assess_strategy
from OracleStrategy import OracleStrategy
from DeepQLearner import Agent, Transition, ReplayMemory

class StockEnvironment:

  def __init__ (self, fixed = 9.95, floating = 0.005, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash
    self.DQN = None
    self.lastBuy = None
    self.sOld = None

    #MACD percentile buckets
    self.macdQ1 = 0
    self.macdQ2 = 0
    self.macdQ3 = 0
    self.macdQ4 = 0

    #BBP
    self.bbpQ1 = 0
    self.bbpQ2 = 0
    self.bbpQ3 = 0
    self.bbpQ4 = 0

    #RSI
    self.rsiQ1 = 0
    self.rsiQ2 = 0
    self.rsiQ3 = 0
    self.rsiQ4 = 0


  def prepare_world (self, start_date, end_date, symbol):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """

    prices = get_data(start_date, end_date, [symbol])
    rsi = RSI(prices, 2)
    macd, signal, histogram = MACD(start_date, end_date, prices)
    bbp = BBP(start_date, end_date, prices, 14)
    prices = prices.join(rsi, how='left')
    prices = prices.join(bbp, how='left')
    prices = prices.join(macd, how='left')
    prices = prices.join(signal, how='left')
    del prices['SIG']
    index_with_nan = prices.index[prices.isnull().any(axis=1)]
    prices.drop(index_with_nan,0, inplace=True)

    #MACD percentile buckets
    self.macdQ1 = np.nanpercentile(prices['MACD'], 20)
    self.macdQ2 = np.nanpercentile(prices['MACD'], 40)
    self.macdQ3 = np.nanpercentile(prices['MACD'], 60)
    self.macdQ4 = np.nanpercentile(prices['MACD'], 80)

    #BBP
    self.bbpQ1 = np.nanpercentile(prices['BBP'], 20)
    self.bbpQ2 = np.nanpercentile(prices['BBP'], 40)
    self.bbpQ3 = np.nanpercentile(prices['BBP'], 60)
    self.bbpQ4 = np.nanpercentile(prices['BBP'], 80)

    #RSI
    self.rsiQ1 = np.nanpercentile(prices['RSI'], 20)
    self.rsiQ2 = np.nanpercentile(prices['RSI'], 40)
    self.rsiQ3 = np.nanpercentile(prices['RSI'], 60)
    self.rsiQ4 = np.nanpercentile(prices['RSI'], 80)
    #print(prices)
    return prices

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def getState(self, data, day, holdings):
    dayData = data.iloc[day].tolist()
    dayData.append(holdings)
    output = []
    for item in dayData[1:]:
      output.append(self.sigmoid(item))
    #print(dayData)
    return output

  def reward(self, day, wallet, sold):

    #Short term reward
    r = wallet.loc[day, 'Value'] - wallet.shift(periods=1).loc[day,'Value']
    '''
    Return short term reward plus the value of sold.
    sold will be the cumulative value of the last long or short position
    or 0 if the position is still open or we are flat
    '''
    #print("Short term reward: " + str(r))
    #print("Reward for selling: " + str(sold))
    return r + sold  #scale this?

  def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0 ):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Print a summary result of what happened at the end of each trip.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """

    print("Initializing Learner and Preparing World...")
    self.DQN = Agent()
    print(start, end, symbol)
    data = self.prepare_world(start, end, symbol)
    print(data)
    wallet = pd.DataFrame(columns=['Cash', 'Holdings', 'Value', 'Trades'], index=data.index)
    #print(data)
    prevSR = 0
    endCondition = False
    tripNum = 0

    #for plotting
    srVals = []
    tVals = []
    losses = []
    aveLoss = 0
    prevFinalVal = 0

    while ((endCondition != True) and (tripNum < 10)):

      wallet['Cash'] = self.starting_cash
      wallet['Holdings'] = 0
      wallet['Value'] = 0
      wallet['Trades'] = 0
      #print(wallet)
      #time.sleep(5)

      firstDay = data.index[0]
      s = self.getState(data, 0, 0)
      a = self.DQN.act(s)
      #break

      print("first action: " + str(a))
      nextTrade = 0
      if (a == 0): #LONG
        nextTrade = 1000
        self.lastBuy = firstDay
      elif (a == 1): #FLAT
        nextTrade = 0
      elif (a == 2): #SHORT
        nextTrade = -1000
        self.lastBuy = firstDay

      #print(nextTrade)
      cost = 0
      if nextTrade != 0:
        cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[firstDay, symbol])

      wallet.loc[firstDay, 'Cash'] -= ((data.loc[firstDay, symbol] * nextTrade) + cost)
      wallet.loc[firstDay, 'Holdings'] += nextTrade
      wallet.loc[firstDay, 'Value'] = wallet.loc[firstDay, 'Cash'] + (data.loc[firstDay, symbol] * wallet.loc[firstDay, 'Holdings'])
      wallet.loc[firstDay, 'Trades'] = nextTrade
      
      #print(wallet)
      sold = 0
      dayCounter = 1
      for day in data.index[1:]:
        #update wallet with yesterdays values
        wallet.loc[day, 'Holdings'] = wallet.shift(periods=1).loc[day, 'Holdings']
        wallet.loc[day, 'Cash'] = wallet.shift(periods=1).loc[day, 'Cash']
        wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])

        self.sOld = s
        #print("Prior State: " + str(s))
        s = self.getState(data, dayCounter, wallet.loc[day, 'Holdings'])
        #print("Current State: " + str(s))
        r = self.reward(day, wallet, sold)
        self.DQN.memory.push(self.sOld, a, s, r)
        #print("Reward: " + str(r))
        #print("sold: " + str(sold))
        loss = self.DQN.train(self.sOld, s, r)
        aveLoss += loss
        
        a = self.DQN.act(s)
        #print("Action: " + str(a))
        ##print(wallet)
        sold = 0
        nextTrade = 0
        if ((a == 0) and (wallet.loc[day, 'Holdings'] != 1000)): #LONG
          #print('buying or holding long position...')
          nextTrade = 1000 - wallet.loc[day, 'Holdings']
          if (wallet.loc[day, 'Holdings'] != 0):
            sold = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']
          self.lastBuy = day          
        elif ((a == 2) and (wallet.loc[day, 'Holdings'] != -1000)): #SHORT
          #print('selling or holding short position...')
          nextTrade = -1000 - wallet.loc[day, 'Holdings']
          if (wallet.loc[day, 'Holdings'] != 0):
            sold = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']
          self.lastBuy = day
        elif (a == 1): #FLAT
          #print('moving to flat position...')
          nextTrade = 0 - wallet.loc[day, 'Holdings']
          if (wallet.loc[day, 'Holdings'] != 0):
            sold = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']

        #print("next Trade: " + str(nextTrade))
        cost = 0
        if nextTrade != 0:
          cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[day, symbol])
          #print("cost " + str(cost))
        wallet.loc[day, 'Cash'] -= (data.loc[day, symbol] * nextTrade) + cost
        wallet.loc[day, 'Holdings'] += nextTrade
        #wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])
        wallet.loc[day, 'Trades'] = nextTrade
        dayCounter += 1
        #print(wallet)

      aveLoss = aveLoss / (tripNum+1)
      losses.append(aveLoss)
      # Compose the output trade list.
      trade_list = []
      print(wallet)
      for day in wallet.index:
        if wallet.loc[day,'Trades'] == 2000:
          trade_list.append([day.date(), symbol, 'BUY', 2000])
        elif wallet.loc[day,'Trades'] == 1000:
          trade_list.append([day.date(), symbol, 'BUY', 1000])
        elif wallet.loc[day,'Trades'] == -1000:
          trade_list.append([day.date(), symbol, 'SELL', 1000])
        elif wallet.loc[day,'Trades'] == -2000:
          trade_list.append([day.date(), symbol, 'SELL', 2000])
      
      print("Trip " + str(tripNum+1) + " complete!")
      #print(trade_list)
      trade_df = pd.DataFrame(trade_list, columns=['Date', 'Symbol', 'Direction', 'Shares'])
      trade_df = trade_df.set_index('Date')
      #print(trade_df)
      trade_df.to_csv('trades.csv')
      #print(wallet)
      #print(trade_df)
      #make call to backtester here
      #stats = assess_strategy(fixed_cost=0, floating_cost=0)
      # if (stats[0] == prevSR):
      #   endCondition = True
      # prevSR = stats[0]
      # srVals.append(stats[4])
      finalVal = wallet['Value'].iloc[-1]
      #srVals.append(finalVal)
      tVals.append(tripNum)
      #if (finalVal == prevFinalVal):
      #  endCondition = True
      prevFinalVal = finalVal
      if (tripNum % 2 == 0):
        self.DQN.tNet.load_state_dict(self.DQN.pNet.state_dict())
        #plt.plot((wallet['Value'] / wallet['Value'].iloc[0]), label=str(tripNum))
      tripNum+=1
      #break
    stats = assess_strategy(fixed_cost=0, floating_cost=0)
    #plt.plot(tVals, srVals)
    plt.plot(tVals, losses, label="Loss")
    plt.title("LossVsEpochs")
    plt.savefig('lossOverTime.png')
    return True

  
  def test_learner( self, start = None, end = None, symbol = None):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Print a summary result of what happened during the test.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000
    """
    print("\nTesting Q Learner from " + str(start) + " to " + str(end) + "\n")
    data = self.prepare_world(start, end, symbol)
    #self.DQN = Agent()
    print(data.to_string())
    wallet = pd.DataFrame(columns=['Cash', 'Holdings', 'Value', 'Trades'], index=data.index)
    firstDay = data.index[0]

    wallet['Cash'] = self.starting_cash
    wallet['Holdings'] = 0
    wallet['Value'] = 0
    wallet['Trades'] = 0

    s = self.getState(data, 0, 0)
    a = self.DQN.testAct(s)

    #print("first action: " + str(a))
    nextTrade = 0
    if (a == 0): #LONG
      nextTrade = 1000
    elif (a == 1): #FLAT
      nextTrade = 0
    elif (a == 2): #SHORT
      nextTrade = -1000

    cost = 0
    if nextTrade != 0:
      cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[firstDay, symbol])
    wallet.loc[firstDay, 'Cash'] -= ((data.loc[firstDay, symbol] * nextTrade) + cost)
    wallet.loc[firstDay, 'Holdings'] += nextTrade
    wallet.loc[firstDay, 'Value'] = wallet.loc[firstDay, 'Cash'] + (data.loc[firstDay, symbol] * wallet.loc[firstDay, 'Holdings'])
    wallet.loc[firstDay, 'Trades'] = nextTrade
    
    dayCounter = 1
    for day in data.index[1:]:
      #update wallet with sOlds values
      wallet.loc[day, 'Holdings'] = wallet.shift(periods=1).loc[day, 'Holdings']
      wallet.loc[day, 'Cash'] = wallet.shift(periods=1).loc[day, 'Cash']
      wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])

      s = self.getState(data, dayCounter, wallet.loc[day, 'Holdings'])
      a = self.DQN.testAct(s)
      #print("Action: " + str(a))
      #print(wallet)
      nextTrade = 0
      if ((a == 0) and (wallet.loc[day, 'Holdings'] != 1000)): #LONG
        #print('buying or holding long position...')
        nextTrade = 1000 - wallet.loc[day, 'Holdings']
      elif ((a == 2) and (wallet.loc[day, 'Holdings'] != -1000)): #SHORT
        #print('selling or holding short position...')
        nextTrade = -1000 - wallet.loc[day, 'Holdings']
      elif (a == 1): #FLAT
        #print('moving to flat position...')
        nextTrade = 0 - wallet.loc[day, 'Holdings']

      #print("next Trade: " + str(nextTrade))
      cost = 0
      if nextTrade != 0:
        cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[day, symbol])
        #print("cost " + str(cost))
      wallet.loc[day, 'Cash'] -= (data.loc[day, symbol] * nextTrade) + cost
      wallet.loc[day, 'Holdings'] += nextTrade
      #wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])
      wallet.loc[day, 'Trades'] = nextTrade
      dayCounter+=1

    trade_list = []

    #print(wallet.to_string())
    for day in wallet.index:
      if wallet.loc[day,'Trades'] == 2000:
        trade_list.append([day.date(), symbol, 'BUY', 2000])
      elif wallet.loc[day,'Trades'] == 1000:
        trade_list.append([day.date(), symbol, 'BUY', 1000])
      elif wallet.loc[day,'Trades'] == -1000:
        trade_list.append([day.date(), symbol, 'SELL', 1000])
      elif wallet.loc[day,'Trades'] == -2000:
        trade_list.append([day.date(), symbol, 'SELL', 2000])

    trade_df = pd.DataFrame(trade_list, columns=['Date', 'Symbol', 'Direction', 'Shares'])
    trade_df = trade_df.set_index('Date')
    trade_df.to_csv('trades.csv')
    #make call to backtester here
    print("Q Trader preformance: ")
    stats = assess_strategy(fixed_cost=0, floating_cost=0)
    o = OracleStrategy(start, end, [symbol])
    print("\nBaseline: ")
    bline = o.test(start_date=start, end_date=end, symbol=symbol)
    print(bline)
    print(wallet)

    plt.clf()
    plt.plot(data.index, wallet['Value'])
    plt.plot(data.index, bline[symbol][25:])
    plt.savefig('TestPerformanceVsBaseline')
    return True
  

if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.

  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.0, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.0', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='WAV_1', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()
  print(args)
  # Create an instance of the environment class.
  env = StockEnvironment( floating=0, fixed=0, starting_cash = args.cash,
                          share_limit = args.shares )
  #floating=0, fixed=0,
  #o = OracleStrategy()
  #bline = o.test(args.train_start, args.train_end)
  #plt.plot(bline / bline.iloc[0])
  print("Beginning training...")
  env.train_learner(args.train_start, args.train_end, args.symbol)
  env.test_learner(args.train_start, args.train_end, args.symbol)
  # Construct, train, and store a Q-learning trader.

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )

