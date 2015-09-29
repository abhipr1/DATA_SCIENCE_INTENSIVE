
# coding: utf-8

# In[139]:

get_ipython().magic('matplotlib inline')
from data_handler import *
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC


# In[97]:

start_date='2000-01-01'
end_date='2015-09-24'


# In[98]:

symbols=['YAHOO/INDEX_DJI','NSE/CNX_NIFTY', 
        'YAHOO/INDEX_HSI', 'YAHOO/INDEX_N225', 
        'YAHOO/INDEX_AXJO', 'YAHOO/SS_000001',
        '^FTSE', '^IXIC', '^DJI']


# In[99]:

def load_dataset():
    #download_and_save_data(symbols, start_date, end_date)
    nifty=get_date_from_file('./Data/CNX_NIFTY_'+start_date+'_'+end_date+'.csv')
    hsi=get_date_from_file('./Data/INDEX_HSI_'+start_date+'_'+end_date+'.csv')
    japan=get_date_from_file('./Data/INDEX_N225_'+start_date+'_'+end_date+'.csv')
    aus=get_date_from_file('./Data/INDEX_AXJO_'+start_date+'_'+end_date+'.csv')
    sanghai=get_date_from_file('./Data/SS_000001_'+start_date+'_'+end_date+'.csv')
    london=get_date_from_file('./Data/FTSE_'+start_date+'_'+end_date+'.csv')
    dow=get_date_from_file('./Data/INDEX_DJI_'+start_date+'_'+end_date+'.csv')
    nasdqa=get_date_from_file('./Data/IXIC_'+start_date+'_'+end_date+'.csv')
    
    return [nifty, hsi, japan, aus, sanghai, london, dow, nasdqa ]


# In[100]:

def daily_return(data):
    data['Daily Return']=data['AdjClose'].pct_change()
    return data


# In[101]:

datasets = load_dataset()
datasets[0].head()


# In[102]:

datasets[0].head(5)


# In[103]:

def add_features(dataframe, adjclose, returns, n):
    return_n = adjclose[9:] + "_Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "_MA" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)


# In[104]:

def moving_average_and_delayed_returns(datasets, delta):
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            add_features(dataset, adjclose, returns, n)
        #print(dataset.head())    
    #return datasets    


# In[105]:

moving_average_and_delayed_returns(datasets, [5,20])


# In[106]:




# In[107]:

datasets[0].Return_CNX_NIFTY.plot()


# In[108]:

def merge_data_frames(datasets, index):
    subset = []
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    return datasets[0].iloc[:, index:].join(subset, how = 'outer')
   


# In[109]:

merged_data=merge_data_frames(datasets, 5)
merged_data.head(2)


# In[110]:

#merged_data=merged_data.index
merged_data.index = pd.to_datetime(merged_data.index)
merged_data.head(2)


# In[111]:

def count_missing(dataframe):
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()


# In[112]:

print ('Size of data frame: %s' %str(merged_data.shape))
print ('Number of NaN after merging: %s' %str(count_missing(merged_data)))
    
merged_data = merged_data.interpolate(method='time')
print ('Number of NaN after time interpolation: %s' % str(count_missing(merged_data)))


# In[113]:

merged_data = merged_data.fillna(merged_data.mean())
print ('Number of NaN after mean interpolation: %s' %count_missing(merged_data)) 


# In[114]:

merged_data.head()


# In[115]:

merged_data.Return_CNX_NIFTY = merged_data.Return_CNX_NIFTY.shift(-1)


# In[116]:

merged_data.head(2)


# In[117]:

def prepare_data_for_classification(dataset, start_test):
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_CNX_NIFTY']
    
    dataset.UpDown[dataset.UpDown >= 0] = 1
    dataset.UpDown[dataset.UpDown < 0] = -1
    
    dataset['UpDown'].fillna(1, inplace=True)    
    
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]    
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test    


# In[118]:

start_test = datetime.datetime(2015,1,1)
X_train, y_train, X_test, y_test  = prepare_data_for_classification(merged_data, start_test)


# In[121]:

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[138]:

X_train.iloc[:,[8,9,10]].plot()


# In[140]:

clf = SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy for SVM Classifier %s" %accuracy)


# In[141]:

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy for KNN Classifier %s" %accuracy) 


# In[142]:

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy for RF Classifier %s" %accuracy) 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



