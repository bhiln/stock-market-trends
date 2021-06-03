import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr

highGrowthStockDf = pd.read_csv('highGrowthStockExample.csv')

print(highGrowthStockDf['TickerSymbol'])

stock 