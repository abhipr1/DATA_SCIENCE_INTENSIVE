__author__ = 'divya'

import pandas as pd
import pandas.io.data
import math
import matplotlib.pyplot as plt
import os

class DataHandler:
    def __init__(self):
        pass

    def fetch_data_from_yahoo(self, symbol, start, end):
        df =  pandas.io.data.get_data_yahoo(symbol, start, end)
        return df

    def fetch_and_save_data(self, symbols, names, start, end):
        ret_df=[]
        for symbol, name in zip(symbols, names):
            file_name='./Data/'+name+'_'+start+'_'+end+'.csv'
            if os.path.exists(file_name):
                #print('[INFO] File(',file_name,') Exists locally')
                df=pd.read_csv(file_name, index_col=0, parse_dates=True)
                ret_df.append(df)
            else:
                df=self.fetch_data_from_yahoo(symbol, start, end)
                df.to_csv(file_name,mode='w')
                ret_df.append(df)
        return ret_df

    def load_data(self, symbols, names, start, end):
        ret_df=[]
        for symbol, name in zip(symbols, names):
            df=pd.read_csv('./Data/'+name+'_'+start+'_'+end+'.csv', index_col=0, parse_dates=True)
            ret_df.append(df)
        return ret_df

    def daily_return(self, data_set):
        data_set['Daily Return'] = data_set['Adj Close'].pct_change()
        return data_set

    def plot_data(self, data_frames, names, subplot=False, market_names=[]):
        row = math.ceil((len(data_frames)*1.0)/2)
        if subplot:
            plt.figure((1), figsize=(80,30))
        i=1
        for market_name, data_frame in zip(market_names, data_frames):
            if subplot:
                plt.subplot(row, 2, i)
            if(len(names)==1):
                data_frame[names].plot(ax=plt.gca())
            else:
                df=[]
                for name in names:
                    df.append(data_frame[name])
                d = pd.concat(df, axis=1)
                d.plot()
                plt.title(name+' For '+market_name.upper())
                plt.legend(loc='best')
            i += 1


