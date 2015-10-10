
import  pandas as pd
import matplotlib.pyplot as plt

def plot(data_frames, s='', title=''):
    for data_frame in data_frames:
        data_frame[s].plot()
        plt.ylabel(s)
        plt.xlabel('Date')
        plt.title(s + ' For ' + title)
        plt.grid(True)
        plt.show()


def concat(a, b):
    return a+b

def renamme_columns(data_frames, market_names):
    for market_name,data_frame in zip(market_names, data_frames):
        columns = data_frame.columns
        data_frame.rename(columns=lambda x: concat(x, '_'+market_name), inplace=True)

# def merge_data_frames(data_frames, index):
#     return pd.concat([data_frame.ix[:, index:] for data_frame in data_frames], axis=1)

def merge_data_frames(data_frames, index):
    keys=[]
    for data_frame in data_frames:
        keys.extend(data_frame.ix[:, index:].columns.values.tolist())
    #print("Keys ======== (((( ",keys," )))) ===========")
    #return pd.concat(data_frames, axis=0, keys=keys)
    return pd.concat([data_frame.ix[:, index:] for data_frame in data_frames], axis=1)


def count_missing(dataframe):
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

