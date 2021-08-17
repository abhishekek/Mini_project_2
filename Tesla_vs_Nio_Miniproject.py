#!/usr/bin/env python
# coding: utf-8

# # Stock Market Analysis 
# 
# 

# ## Companies - Tesla and Nio
# 
# ## Submitted by - Abhishek Satayavir Sharma
# ## Roll no- 205320002

# # 1. What was the change in price of the stock overtime?
# 
# In this section we'll go over how to handle requesting stock information with pandas, and how to analyze basic attributes of a stock.

# In[223]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime


# In[224]:


# The EV stocks we'll use for this analysis
ev_list = ['TSLA','NIO']
# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
Tesla=DataReader(ev_list[0],'yahoo',start,end);
Nio=DataReader(ev_list[1],'yahoo',start,end);


# In[225]:


# Summary Stats
Tesla.describe()


# In[226]:


# General info
Tesla.head()


# In[227]:


# Let's see a historical view of the closing price
plt.figure(figsize=(15, 16))
#plt.subplots_adjust(top=1.25, bottom=1.2)
plt.subplot(2, 2, 1)
Tesla['Adj Close'].plot()
plt.ylabel('Adj Close',fontsize=10)
plt.xlabel('Date',fontsize=10)
plt.title("Closing Price of Tesla",fontsize=20);


plt.subplot(2,2,2)
Nio['Adj Close'].plot()
plt.ylabel('Adj Close',fontsize=10)
plt.xlabel('Date',fontsize=10)
plt.title(f"Closing Price of Nio",fontsize=20)
    
plt.tight_layout()


# In[228]:


# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(15, 7))
plt.subplots_adjust(top=1.25, bottom=1.2)
plt.subplot(2, 1, 1)
Tesla['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)
plt.title("Sales Volume for Tesla",fontsize=20)
plt.tight_layout()

plt.subplot(2, 1, 2)
Nio['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)
plt.title("Sales Volume for Tesla",fontsize=20)
plt.tight_layout()


# Now that we've seen the visualizations for the closing price and the volume traded each day, let's go ahead and caculate the moving average for the stock.

# # 2. What was the moving average of the two stocks?

# In[229]:


ma_day = [10, 20, 50,100]

for ma in ma_day:
    column_name = f"MA for {ma} days"
    Tesla[column_name] = Tesla['Adj Close'].rolling(ma).mean()
    Nio[column_name] = Nio['Adj Close'].rolling(ma).mean()


# Now let's go ahead and plot all the additional Moving Averages

# In[230]:


# df.groupby("company_name").hist(figsize=(12, 12));


# In[231]:


plt.figure(figsize=(15,6))
plt.plot(Tesla[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days','MA for 100 days']])
plt.legend(['Current price','Mean for 10 days','Mean for 20 days','Mean for 50 days','Mean for 100 days'])
plt.title('Tesla')
plt.show()


# In[232]:


plt.figure(figsize=(15,6))
plt.plot(Nio[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days','MA for 100 days']])
plt.legend(['Current price','Mean for 10 days','Mean for 20 days','Mean for 50 days','Mean for 100 days'])
plt.title('Nio')
plt.show()


# # 3. What was the daily return of the stock on average?

# Now that we've done some baseline analysis, let's go ahead and dive a little deeper. We're now going to analyze the risk of the stock. In order to do so we'll need to take a closer look at the daily changes of the stock, and not just its absolute value. Let's go ahead and use pandas to retrieve the daily returns for the Tesla and Nio stock.

# In[233]:


# We'll use pct_change to find the percent change for each day
Tesla['Daily Return'] = Tesla['Adj Close'].pct_change()
Nio['Daily Return'] = Nio['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_figheight(8)
fig.set_figwidth(15)

Tesla['Daily Return'].plot(ax=axes[0], legend=True, linestyle='--', marker='o')
axes[0].set_title('Tesla')

Nio['Daily Return'].plot(ax=axes[1], legend=True, linestyle='--', marker='o')
axes[1].set_title('Nio')

fig.tight_layout()


# In[234]:


plt.figure(figsize=(20,10))
Nio['Daily Return'].hist(bins=30,color='violet')
plt.xlabel(None)
plt.ylabel('Daily return',fontsize=20)
plt.title('Nio Daily percentage return',fontsize=20)


# 

# In[235]:


# Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
plt.figure(figsize=(20,10))
sns.distplot(Tesla['Daily Return'].dropna(), bins=100, color='red')
plt.ylabel('Daily Return',fontsize=20)
plt.xlabel(None,fontsize=20)
plt.tight_layout()
plt.title('Daily percetage return of Tesla',fontsize=20)


# # 4. What was the correlation between different stocks closing prices?

# 

# In[236]:


# Grab all the closing prices for the tech stock list into one DataFrame
closing = DataReader(ev_list, 'yahoo', start, end)['Adj Close']

# Let's take a quick look
closing.head() 


# Now we can compare the daily percentage return of two stocks to check how correlated. First let's see a sotck compared to itself.

# In[237]:


# Comparing Tesla to itself should show a perfectly linear relationship
sns.jointplot('TSLA', 'NIO',closing,kind='scatter', color='seagreen')


# In[238]:


sns.pairplot(closing,kind='reg')


# It perfectly make sense that the share price of both the stock are correlated as both belongs to EV domains and the current hype of EV make Nio a multibagger from a penny stock. To show this, let's look the rise of this stock in the Corona Pandemic alone from 16/11/19-16/11/20

# In[239]:


#Plotting the one year data during the pandemic


# In[240]:


end = datetime(2020,11,16)
start = datetime(end.year - 1, end.month, end.day)
Nio_rise=DataReader('NIO','yahoo',start,end);
plt.figure(figsize=(10,10))
Nio_d=Nio_rise.dropna()
plt.plot(Nio_d['Open'])


# In[243]:


plt.figure(figsize=(10,7))
Nio_d['Daily Return'] = Tesla['Close'].pct_change()
plt.plot(Nio_d['Daily Return'],color='red',marker='o',linestyle='--')


# In[244]:


Nio_d['Daily Return'].fillna(0)


# In[245]:


# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(Tesla.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) 
# or the color map (BluePurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
return_fig.map_diag(plt.hist, bins=30)


# In[246]:


#let's see the correlation between each of them
(Tesla.corrwith(Nio))
closing_df = DataReader(ev_list, 'yahoo', start, end)['Adj Close']


# Finally, we could also do a correlation plot, to get actual numerical values for the correlation between the stocks' daily return values. By comparing the closing prices, we see an interesting relationship between Tesla and Nio

# In[247]:


closing_df.head()
tech_rets = closing_df.pct_change()
tech_rets.head()


# In[248]:


sns.pairplot(tech_rets, kind='reg')


# In[249]:


sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
#tech_rets is the percentage change 


# In[250]:


sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
plt.show()


# # 5. How much value do we put at risk by investing in a Tesla?

# There are many ways we can quantify risk, one of the most basic ways using the information we've gathered on daily percentage returns is by comparing the expected return with the standard deviation of the daily returns.

# In[251]:


# Let's start by defining a new DataFrame as a clenaed version of the oriignal Tesla DataFrame
# Let's start by defining a new DataFrame as a clenaed version of the oriignal tech_rets DataFrame
rets = tech_rets.dropna()

area = np.pi * 20

plt.figure(figsize=(10, 7))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))


# ### Therefore, the overall conclusion from this small mini project is that the Nio stock is at high risk and high return than Tesla.
# ### The fundamental and technical analysis of these stocks goes in unison with the results obtain from this exercise
