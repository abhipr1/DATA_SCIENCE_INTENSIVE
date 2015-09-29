
# coding: utf-8

# In[7]:

import Quandl as qd
import pandas as pd
import pandas.io.data
import seaborn


# In[8]:

def fetch_data_from_yahoo(symbol, start, end):
    """
    downloads stock from yahoo
    """
    df =  pandas.io.data.get_data_yahoo(symbol, start, end)
    symbol = symbol.replace('^','')
    
    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()

    return df


# In[9]:

def fetch_data_from_quandl(symbol, start_date, end_date):
    df =  qd.get(symbol, trim_start = start_date, trim_end = end_date, authtoken="rzpJEj6ebQyTiSXF4pS3")
    symbol = symbol[symbol.find('/')+1:]
    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
    return df


# In[10]:

def download_and_save_data(symbols, start_date, end_date):
    for symbol in symbols:
        df = None
        if symbol.find('/') != -1:
            df = fetch_data_from_quandl(symbol, start_date, end_date)
            df.to_csv(path_or_buf='./Data/'+symbol[symbol.find('/')+1:]+'_'+start_date+'_'+end_date+'.csv')
        else:
            print(symbol)
            #df = read_csv(symbol)
            df = fetch_data_from_yahoo(symbol, start_date, end_date)
            df.to_csv(path_or_buf='./Data/'+symbol.replace('^','')+'_'+start_date+'_'+end_date+'.csv')


# In[11]:

def get_date_from_file(file_name):
    df = pd.read_csv(file_name)
    if len(df.columns)>6:
        df.drop(df.columns[[5]], axis=1, inplace=True)
    df.set_index(['Date'], inplace=True)
    return df


# In[12]:

def read_csv(symbol):
    df =  pd.read_csv('./Data/table.csv')
    symbol = symbol.replace('^','')
    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()

    return df


# In[ ]:




# In[ ]:



