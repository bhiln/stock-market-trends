#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Appian Stock Data

# In[2]:


date = '2019-05-31'
appn = pdr.get_data_yahoo('APPN',date)
appn.drop('Adj Close', axis=1, inplace=True)
appn.head()


# In[3]:


appn['3-day'] = appn['Close'].rolling(3).mean()
appn['9-day'] = appn['Close'].rolling(9).mean()
appn['21-day'] = appn['Close'].rolling(21).mean()
appn['Change'] = (appn.Close-appn.Close.shift())/appn.Close.shift()
appn.tail()


# In[4]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Close Moving Average")
plt.plot(appn['Close'],label='Close')
plt.plot(appn['3-day'],label='Moving Average 3 Days')
plt.plot(appn['9-day'],label='Moving Average 9 Days')
plt.plot(appn['21-day'],label='Moving Average 21 Days')
plt.legend(loc=2)


# ## Zoom in the changing period

# In[5]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Zoomed Close Moving Average")
plt.plot(appn['Close'][-200:-100],label='Close')
plt.plot(appn['3-day'][-200:-100],label='Moving Average 3 Days')
plt.plot(appn['9-day'][-200:-100],label='Moving Average 9 Days')
plt.plot(appn['21-day'][-200:-100],label='Moving Average 21 Days')
plt.legend(loc=1)


# In[6]:


# When the faster moving average is moving above the slower one, indicates time to buy
appn['position-9-21'] = np.where(appn['9-day'] > appn['21-day'], 1,0)
# When the faster moving average is moving below the slower one, indicates time to sell
appn['position-9-21'] = np.where(appn['9-day'] < appn['21-day'],-1,appn['position-9-21'])


# In[7]:


plt.title("9-21 Buy/Sell Activity")
appn['position-9-21'].plot(figsize=(10,6))


# In[8]:


# Whether we are making money with this system
appn['system-9-21'] = np.where(appn['position-9-21'] > 0, appn['position-9-21']* appn['Change'],0)
appn[['Change','system-9-21']].cumsum().plot(figsize=(10,6))
plt.title("9-21 Moving Average Performance vs. Buy/Hold Performance")


# In[9]:


# When the faster moving average is moving above the slower one, indicates time to buy
appn['position-3-9'] = np.where(appn['3-day'] > appn['9-day'], 1,0)
# When the faster moving average is moving below the slower one, indicates time to sell
appn['position-3-9'] = np.where(appn['3-day'] < appn['9-day'],-1,appn['position-3-9'])
appn['position-3-9'].plot(figsize=(10,6))
plt.title("3-9 Buy/Sell Activity")


# In[10]:


# Whether we are making money with this system
appn['system-3-9'] = np.where(appn['position-3-9'] > 0, appn['position-3-9']* appn['Change'],0)
appn[['Change','system-3-9']].cumsum().plot(figsize=(10,6))
plt.title("3-9 Moving Average Performance vs. Buy/Hold Performance")


# ## Prepare data for DGIM processing

# In[11]:


bit_depth = 16
num_buckets = 5
quiet = True
appn_list = appn['Close'].tolist()
appn_3day_list = appn['3-day'].tolist()
appn_9day_list = appn['9-day'].tolist()
appn_21day_list = appn['21-day'].tolist()
appn_list_bin = [ np.array(list(np.binary_repr(round(elem)).zfill(bit_depth))).astype(np.int16) for elem in appn_list ]
print(appn_list[0],appn_list_bin[0], len(appn_list_bin))


# In[12]:


from DGIMMovingAverage import *


# ## 3, 9, 21 day moving average with DGIM

# In[13]:


dgim_3day = DGIMMovingAverage(appn_list_bin, 3, num_buckets, bit_depth, appn_3day_list, quiet)
appn['3-day-dgim'] = dgim_3day.mov_avg
appn.tail()

plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("3 Day DGIM Moving Average Performance vs. 3 Day Moving Average Performance")
plt.plot(range(len(dgim_3day.mov_avg)), dgim_3day.mov_avg, label = "DGIM 3 day")
plt.plot(range(len(dgim_3day.mov_avg)), appn_3day_list, label = "APPN 3 day")
plt.legend(loc=1)

plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("3 Day DGIM Moving Average Error")
plt.plot(range(len(dgim_3day.error)), dgim_3day.error, label = "DGIM 3day error")
plt.legend(loc=1)


# In[14]:


dgim_9day = DGIMMovingAverage(appn_list_bin, 9, num_buckets, bit_depth, appn_9day_list, quiet)
appn['9-day-dgim'] = dgim_9day.mov_avg
appn.tail()

plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("9 Day DGIM Moving Average Performance vs. 9 Day Moving Average Performance")
plt.plot(range(len(dgim_9day.mov_avg)), dgim_9day.mov_avg, label = "DGIM 9 day")
plt.plot(range(len(dgim_9day.mov_avg)), appn_9day_list, label = "APPN 9 day")
plt.legend(loc=1)

plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("9 Day DGIM Moving Average Error")
plt.plot(range(len(dgim_9day.error)), dgim_9day.error, label = "DGIM 9day error")
plt.legend(loc=1)


# In[15]:


dgim_21day = DGIMMovingAverage(appn_list_bin, 21, num_buckets, bit_depth, appn_21day_list, quiet)
appn['21-day-dgim'] = dgim_21day.mov_avg
appn.tail()

plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("21 Day DGIM Moving Average Performance vs. 21 Day Moving Average Performance")
plt.plot(range(len(dgim_21day.mov_avg)), dgim_21day.mov_avg, label = "DGIM 21 day")
plt.plot(range(len(dgim_21day.mov_avg)), appn_21day_list, label = "APPN 21 day")
plt.legend(loc=1)

plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("21 Day DGIM Moving Average Error")
plt.plot(range(len(dgim_21day.mov_avg)), dgim_21day.error, label = "DGIM 21day error")
plt.legend(loc=1)


# In[16]:


appn.tail()


# ## 9/21 DGIM Comparison

# In[17]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("9-21 DGIM Moving Average Performance vs. 9-21 Moving Average Performance vs. Buy/Hold Performance")
plt.plot(appn['Close'][-200:-100],label='Close')
plt.plot(appn['9-day-dgim'][-200:-100],label='DGIM Moving Average 9 Days')
plt.plot(appn['21-day-dgim'][-200:-100],label='DGIM Moving Average 21 Days')
plt.plot(appn['9-day'][-200:-100],label='Moving Average 9 Days')
plt.plot(appn['21-day'][-200:-100],label='Moving Average 21 Days')
plt.legend(loc=0)


# In[18]:


# When the faster moving average is moving above the slower one, indicates time to buy
appn['position-dgim-9-21'] = np.where(appn['9-day-dgim'] > appn['21-day-dgim'], 1,0)
# When the faster moving average is moving below the slower one, indicates time to sell
appn['position-dgim-9-21'] = np.where(appn['9-day-dgim'] < appn['21-day-dgim'],-1,appn['position-dgim-9-21'])
plt.title("9-21 Buy/Sell Activity")
appn['position-dgim-9-21'].plot(figsize=(10,6))


# In[19]:


# Whether we are making money with this system
appn['system-dgim-9-21'] = np.where(appn['position-dgim-9-21'] > 0, appn['position-dgim-9-21']* appn['Change'],0)
appn[['Change','system-dgim-9-21']].cumsum().plot(figsize=(10,6))
plt.title("9-21 DGIM Moving Average Performance vs. Buy/Hold Performance")
appn[['Change','system-dgim-9-21']].cumsum().tail()


# In[20]:


appn[['system-9-21','system-dgim-9-21']].cumsum().plot(figsize=(10,6))
plt.title("9-21 DGIM Moving Average Performance vs. 9-21 Moving Average Performance")
appn[['system-9-21','system-dgim-9-21']].cumsum().tail()


# In[21]:


appn.tail()


# ## 3/9 DGIM comparison

# In[22]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("3-9 DGIM Moving Average Performance vs. 3-9 Moving Average Performance vs. Buy/Hold Performance")
plt.plot(appn['Close'][-200:-100],label='Close')
plt.plot(appn['3-day-dgim'][-200:-100],label='DGIM Moving Average 3 Days')
plt.plot(appn['9-day-dgim'][-200:-100],label='DGIM Moving Average 9 Days')
plt.plot(appn['3-day'][-200:-100],label='Moving Average 3 Days')
plt.plot(appn['9-day'][-200:-100],label='Moving Average 9 Days')
plt.legend(loc=1)


# In[23]:


# When the faster moving average is moving above the slower one, indicates time to buy
appn['position-dgim-3-9'] = np.where(appn['3-day-dgim'] > appn['9-day-dgim'], 1,0)
# When the faster moving average is moving below the slower one, indicates time to sell
appn['position-dgim-3-9'] = np.where(appn['3-day-dgim'] < appn['9-day-dgim'],-1,appn['position-dgim-3-9'])
plt.title("3-9 Buy/Sell Activity")
appn['position-dgim-3-9'].plot(figsize=(10,6))


# In[24]:


# Whether we are making money with this system
appn['system-dgim-3-9'] = np.where(appn['position-dgim-3-9'] > 0, appn['position-dgim-3-9']* appn['Change'],0)
appn[['Change','system-dgim-3-9']].cumsum().plot(figsize=(10,6))
plt.title("3-9 DGIM Moving Average Performance vs. Buy/Hold Performance")
appn[['Change','system-dgim-3-9']].cumsum().tail()


# In[25]:


appn[['system-3-9','system-dgim-3-9']].cumsum().plot(figsize=(10,6))
plt.title("3-9 DGIM Moving Average Performance vs. 3-9 Moving Average Performance")
appn[['system-3-9','system-dgim-3-9']].cumsum().tail()


# ## Normalizing DGIM

# In[26]:


from sklearn import preprocessing

df3 = appn['3-day-dgim']
df9 = appn['9-day-dgim']
df21 = appn['21-day-dgim']

appn['3-day-dgim-norm']=(df3-df3.mean())/df3.std()
appn['9-day-dgim-norm']=(df9-df9.mean())/df9.std()
appn['21-day-dgim-norm']=(df21-df21.mean())/df21.std()

# normalized_df.plot()
plt.title("Normalized DGIM Moving Average")
appn['3-day-dgim-norm'].plot(figsize=(10,6))
appn['9-day-dgim-norm'].plot(figsize=(10,6))
appn['21-day-dgim-norm'].plot(figsize=(10,6))


# In[27]:


# When the faster moving average is moving above the slower one, indicates time to buy
appn['position-dgim-3-9-norm'] = np.where(appn['3-day-dgim-norm'] > appn['9-day-dgim-norm'], 1,0)
# When the faster moving average is moving below the slower one, indicates time to sell
appn['position-dgim-3-9-norm'] = np.where(appn['3-day-dgim-norm'] < appn['9-day-dgim-norm'],-1,appn['position-dgim-3-9-norm'])
plt.title("Buy/Sell Activity")
appn['position-dgim-3-9-norm'].plot(figsize=(10,6))


# In[28]:


# Whether we are making money with this system
appn['system-dgim-3-9-norm'] = np.where(appn['position-dgim-3-9-norm'] > 0, appn['position-dgim-3-9-norm']* appn['Change'],0)
appn[['Change','system-dgim-3-9-norm']].cumsum().plot(figsize=(10,6))
plt.title("3-9 Normalized DGIM Moving Average Performance vs. Buy/Hold Performance")
appn[['Change','system-dgim-3-9-norm']].cumsum().tail()


# In[29]:


appn[['system-dgim-3-9-norm','system-dgim-3-9']].cumsum().plot(figsize=(10,6))
plt.title("3-9 Normalized DGIM Moving Average Performance vs. 3-9 DGIM Moving Average performance")
appn[['system-dgim-3-9-norm','system-dgim-3-9']].cumsum().tail()


# In[30]:


appn[['system-dgim-3-9-norm','system-3-9']].cumsum().plot(figsize=(10,6))
plt.title("3-9 Normalized DGIM Moving Average Performance vs. 3-9 Moving Average Performance")
appn[['system-dgim-3-9-norm','system-3-9']].cumsum().tail()


# In[31]:


# When the faster moving average is moving above the slower one, indicates time to buy
appn['position-dgim-9-21-norm'] = np.where(appn['9-day-dgim-norm'] > appn['21-day-dgim-norm'], 1,0)
# When the faster moving average is moving below the slower one, indicates time to sell
appn['position-dgim-9-21-norm'] = np.where(appn['9-day-dgim-norm'] < appn['21-day-dgim-norm'],-1,appn['position-dgim-9-21-norm'])
plt.title("Buy/Sell Activity")
appn['position-dgim-9-21-norm'].plot(figsize=(10,6))


# In[32]:


# Whether we are making money with this system
appn['system-dgim-9-21-norm'] = np.where(appn['position-dgim-9-21-norm'] > 0, appn['position-dgim-9-21-norm']* appn['Change'],0)
appn[['Change','system-dgim-9-21-norm']].cumsum().plot(figsize=(10,6))
plt.title("9-21 Normalized DGIM Moving Average Performance vs. Buy/Hold Performance")
appn[['Change','system-dgim-9-21-norm']].cumsum().tail()


# In[33]:


appn[['system-dgim-9-21-norm','system-dgim-9-21']].cumsum().plot(figsize=(10,6))
plt.title("9-21 Normalized DGIM Moving Average Performance vs. 9-21 DGIM Moving Average performance")
appn[['system-dgim-9-21-norm','system-dgim-9-21']].cumsum().tail()


# In[34]:


appn[['system-dgim-9-21-norm','system-9-21']].cumsum().plot(figsize=(10,6))
plt.title("9-21 Normalized DGIM Moving Average Performance vs. 9-21 Moving Average Performance")
appn[['system-dgim-9-21-norm','system-9-21']].cumsum().tail()


# In[35]:


appn[['system-dgim-3-9-norm','system-dgim-9-21-norm']].cumsum().plot(figsize=(10,6))
plt.title("3-9 Normalized DGIM Moving Average Performance vs. 9-21 Normalized DGIM Moving Average Performance")
appn[['system-dgim-3-9-norm','system-dgim-9-21-norm']].cumsum().tail()


# ## Generate USD Gain

# In[36]:


sys39 = [appn['Close'][0]]
sys921 = [appn['Close'][0]]
sysdgimnorm39 = [appn['Close'][0]]
sysdgimnorm921 = [appn['Close'][0]]

for i in range(1,len(appn["Close"])):
    sys39.append(sys39[i-1] * (1+appn["system-3-9"][i]))
    sys921.append(sys921[i-1] * (1+appn["system-9-21"][i]))
    sysdgimnorm39.append(sysdgimnorm39[i-1] * (1+appn["system-dgim-3-9-norm"][i]))
    sysdgimnorm921.append(sysdgimnorm921[i-1] * (1+appn["system-dgim-9-21-norm"][i]))

appn["system-3-9-usd"] = sys39
appn["system-9-21-usd"] = sys921
appn["system-dgim-3-9-norm-usd"] = sysdgimnorm39
appn["system-dgim-9-21-norm-usd"] = sysdgimnorm921    

appn[['Close','system-3-9-usd','system-9-21-usd','system-dgim-3-9-norm-usd','system-dgim-9-21-norm-usd']].fillna(method ='pad').plot(title="APPN USD",figsize=(20,12))
plt.ylabel('US Dollars ($)')
plt.title("USD Gains for each method")

appn[['Close','system-3-9-usd','system-9-21-usd','system-dgim-3-9-norm-usd','system-dgim-9-21-norm-usd']].tail()


# ## Winning Method: 3-9 Normalized DGIM Moving Average
# ### Runner Up: 3-9 Moving Average

# In[37]:


appn[['Change','system-dgim-3-9-norm','system-3-9','system-dgim-9-21-norm','system-9-21']].cumsum().plot(figsize=(20,12))
plt.title("Method Performance")
appn[['Change','system-dgim-3-9-norm','system-3-9','system-dgim-9-21-norm','system-9-21']].cumsum().tail()


# In[ ]:

plt.show()


