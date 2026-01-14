#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%pip install yfinance


# In[17]:


import yfinance as yfin
import os


start = '1950-01-01'


# In[ ]:


def download_price_data(ticker_list_file, start_date, end_date, dest_dir):
    # create dest_dir if not exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(ticker_list_file, 'r') as f:
        tickers = f.readlines()
        # replace space to dash '-'
        tickers = [ticker.replace(' ', '-').strip() for ticker in tickers]
    for ticker in tickers:
        try:
            data = yfin.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False, timeout=40)
            data.index = data.index.date.astype(str)
            data.to_csv(f'{dest_dir}/{ticker}.csv', index_label='Date')
            print(f'Downloaded data for {ticker}')
        except Exception as e:
            print(f'Failed to download data for {ticker}: {e}')
            continue



# In[19]:


download_price_data('./data/sp_500.txt', start, None, './data/prices')


# In[ ]:




