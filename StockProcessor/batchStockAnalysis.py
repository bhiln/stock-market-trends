import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr

highGrowthStockDf = pd.read_csv('highGrowthStockExample.csv')
recommendedStockDf = pd.read_csv('recommendedStock.csv')

targetDf = recommendedStockDf

twoYearGainList =[]

print(targetDf['TickerSymbol'])

for i in range(len(targetDf.index)):

  stock_name = targetDf.loc[i,'TickerSymbol']
  # print(stock_name)

  # Download data from Yahoo Finance API
  stock = pdr.get_data_yahoo(stock_name,'2019-05-31')

  # Dropping Unused Columns
  stock.drop('Adj Close', axis=1, inplace=True)
  stock.drop('High', axis=1, inplace=True)
  stock.drop('Low', axis=1, inplace=True)
  stock.drop('Open', axis=1, inplace=True)
  stock.drop('Volume', axis=1, inplace=True)
  print(stock.head())

  # Get the Moving Averages for 3-day, 9-day and 21-day
  stock['3-day'] = stock['Close'].rolling(3).mean()
  stock['9-day'] = stock['Close'].rolling(9).mean()
  stock['21-day'] = stock['Close'].rolling(21).mean()

  # Get the daily changes
  stock['Change'] = np.log(stock.Close / stock.Close.shift())

  # 3-day vs 9-day MA.
  # Register 1 when 3-day MA is greater than 9-day MA and -1 when 3-day is less than 9-day
  stock['Position_3v9'] = np.where(stock['3-day'] > stock['9-day'], 1, 0)
  stock['Position_3v9'] = np.where(stock['3-day'] < stock['9-day'], -1, stock['Position_3v9'])

  # 9-day vs 21-day MA.
  # Register 1 when 9-day MA is greater than 21-day MA and -1 when 9-day is less than 21-day
  stock['Position_9v21'] = np.where(stock['9-day'] > stock['21-day'], 1,0)
  stock['Position_9v21'] = np.where(stock['9-day'] < stock['21-day'],-1,stock['Position_9v21'])

  # Calculate the volatility of the stock
  stock['Volatility_3d'] = stock.Change.rolling(3).std().shift()

  stock['System_3v9'] = np.where(stock['Position_3v9'] > 0, stock['Position_3v9'] * stock['Change'], 0)
  print(stock['System_3v9'].cumsum().tail(1)[0])
  gain = stock['System_3v9'].cumsum().tail(1)[0]
  change = stock['Change'].cumsum().tail(1)[0]
  

  twoYearGainList.append({ 'StockName': stock_name, "Gain": gain, "Change": change })


print("two year gain list is: ")
print(twoYearGainList)

twoYearGainDf = pd.DataFrame(twoYearGainList)
print(twoYearGainDf)

meanDifference = (twoYearGainDf['Gain'] - twoYearGainDf['Change']).mean()
print(meanDifference)