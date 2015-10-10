from DataHandler import *
from Util import *
from DataProcessor import *
from Portfolio import *


from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_date(dat, end_date_intervel, start_date_intervel):
    e = (datetime.strptime(dat, '%Y-%m-%d')+ relativedelta(days=end_date_intervel)).strftime('%Y-%m-%d')
    s = (datetime.strptime(dat, '%Y-%m-%d')+ relativedelta(days=start_date_intervel)).strftime('%Y-%m-%d')
    return  e, s

start_date_list = ['2014-01-01', '2012-01-01', '2013-05-01', '2015-07-01', '2010-02-01', '2011-11-15'
    , '2015-02-01','2015-05-01','2015-06-01','2015-07-01', '2015-08-01']
end_date_list   = [450, 60, 60, 90, 75, 90, 60, 60, 60, 60, 60]
start_test_list = [330, 45, 45, 75, 60, 80, 50, 45, 45, 45, 45]

#indices = ['^NSEI', '^DJI', '^FTSE', '^AXJO', '^HSI', '^N225', '^IXIC']#, '000001.SS']
#market_name = ['nse', 'dji', 'ftse', 'aus', 'hsi', 'nikkei', 'nasdaq']#, 'sanghai']

indices = ['^NSEI', '^BSESN', '^AXJO', '^HSI', '^N225']#, '000001.SS']
market_name = ['nse', 'bse', 'aus', 'hsi', 'nikkei']#, 'sanghai']



if __name__ == '__main__':
    final_result=''
    for start_date,e,s in zip(start_date_list, end_date_list, start_test_list):
        end_date, start_test = get_date(start_date, e,s)

        d = DataHandler()
        data_frames = d.fetch_and_save_data(indices, market_name, start_date, end_date)

        # plot([data_frames[0]], 'Adj Close', market_name[0])

        for data_frame in data_frames:
            d.daily_return(data_frame)

        # for index, data_frame in zip(range(len(data_frames)), data_frames):
        #     plot([d.daily_return(data_frame)], 'Daily Return', market_name[index].upper()+' Index')

        # print(data_frames[0].head(5))

        #d.plot_data([data_frames[0],data_frames[7]], ['Daily Return'], market_names=market_name)
        #plt.show()

        # Plot data
        # data_frames[0]['Daily Return'].plot()

        #print(data_frames[0].index.name)
        #print(data_frames[0].columns.values.tolist())

        dp = DataProcessor()

        data_points = [4, 8, 12]

        # Compute moving average
        for data_frame in data_frames:
            data_frame = dp.get_moving_average(data_frame, data_points)

        #Compute exponential moving average
        for data_frame in data_frames:
            data_frame = dp.get_ewma(data_frame, data_points)

        # cols=['Adj Close','MA_5','MA_10','MA_15','MA_20']
        # plot different calculations

        '''
        for i in range(2):
            data_frames[i]['Adj Close'].plot(legend=True, linestyle='-', linewidth=2)
            cols=[ 'MA_5']
            for col in cols:
                data_frames[i][col].plot(legend=True, linestyle='-', linewidth=2)
                plt.grid(linestyle='--', linewidth=2)
                plt.title(market_name[i].upper()+' ADj Close & MA')


            cols_ema=['EMA_5','EMA_10']
            for col in cols_ema:
                data_frames[i][col].plot(legend=True, linewidth=2)
                plt.title(market_name[i].upper()+' ADj Close MA 20 & EMA 20')


            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

        '''
        renamme_columns(data_frames, market_name)
        #print(data_frames[1].index.name)
        #print([data_frame.columns for data_frame in data_frames])

        # for name, data_frame in zip(market_name, data_frames):
        #     print("No of Data for [%8s] are [%s]" % (name.upper(), len(data_frame)))

        merged_data = merge_data_frames(data_frames, 5)
        #print("============", merged_data.columns, "===================")
        #print(merged_data.describe())

        print(merged_data.columns.values.tolist())

        merged_data.Return_CNX_NIFTY = merged_data['Daily Return_nse'].shift(-1)

        # Plot is broken due to missing data
        # merged_data.Return_CNX_NIFTY.plot()

        # merged_data['Adj Close_nse'].plot()
        # plt.show()

        print("Shape of merged data", merged_data.shape, ".")
        print("After merge out of [", len(merged_data) * len(merged_data.columns), "] [", count_missing(merged_data),
              "] data points are missing.")

        #print("Merged data Index = ", merged_data.index)
        merged_data = merged_data.interpolate(method='time')
        print('Number of NaN after time interpolation: %s' % str(count_missing(merged_data)))

        merged_data = merged_data.fillna(merged_data.mean())
        print('Number of NaN after mean interpolation: %s' % count_missing(merged_data))

        # Plot after
        # merged_data['Adj Close_nse'].plot()

        # merged_data['Daily Return_nse'].plot()
        # plt.show()

        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2)
        # pca.fit(merged_data)

        X_train, y_train, X_test, y_test = dp.prepare_data_for_classification(merged_data, start_test)

        # print("======== Shapes ======== ")
        # print("Training X", X_train.shape)
        # print("Training y", y_train.shape)
        # print("Test     X", X_test.shape)
        # print("Test     y", y_test.shape)
        # print("======================== ")

        # plt.figure()
        # y_test.plot(kind='bar', alpha=0.5)
        # plt.title('Test data plot')
        # plt.axhline(0, color='k')
        # plt.show()
        #
        # plt.figure()
        # y_train.plot(kind='bar', alpha=0.9)
        # plt.axhline(0, color='r')
        # plt.title('Train data plot')
        # plt.show()

        print("Positive and negative movement in train data outcome.")
        print(y_train.value_counts())
        print("Positive and negative movement in test data outcome.")
        print(y_test.value_counts())

        dp.apply_logistic_regressor(X_train, y_train, X_test, y_test)
        dp.apply_svc(X_train, y_train, X_test, y_test)
        dp.apply_knn(X_train, y_train, X_test, y_test)
        dp.apply_random_forest(X_train, y_train, X_test, y_test)

        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
        bp=dp.select_best_param_svc(X_train, y_train, parameters)
        dp.apply_svc(X_train, y_train, X_test, y_test, kernel='rbf', C=1)
        dp.apply_svc(X_train, y_train, X_test, y_test, kernel='linear', C=1)


        symbol = 'CNX-NIFTY'
        bars = d.fetch_data_from_yahoo('^NSEI', start_test, end_date)

        X_train, y_train, X_test, y_test = dp.partition_data(merged_data, len(bars))

        predict_svc = dp.get_svc_prediction(X_train, y_train, X_test, kernel=bp['kernel'], C=bp['C'])
        signals_svc = pd.DataFrame(index=bars.index)
        signals_svc['signal'] = 0.0
        signals_svc['signal'] = predict_svc
        signals_svc['positions'] = signals_svc['signal'].diff()
        portfolio_svc = MarketIntradayPortfolio(symbol, bars, signals_svc)
        returns_svc = portfolio_svc.backtest_portfolio()


        predict_rf = dp.get_randomforest_prediction(X_train, y_train, X_test, 50)
        signals_rf = pd.DataFrame(index=bars.index)
        signals_rf['signal'] = 0.0
        signals_rf['signal'] = predict_rf
        signals_rf['positions'] = signals_rf['signal'].diff()


        portfolio_rf = MarketIntradayPortfolio(symbol, bars, signals_rf)
        returns_rf = portfolio_rf.backtest_portfolio()
        # print(signals_rf)
        #
        # print(returns_rf)
        # print(returns_svc)

        predict_lr = dp.get_logistic_reg_prediction(X_train, y_train, X_test)
        signals_lr = pd.DataFrame(index=bars.index)
        signals_lr['signal'] = 0.0
        signals_lr['signal'] = predict_lr
        signals_lr['positions'] = signals_lr['signal'].diff()
        portfolio_lr = MarketIntradayPortfolio(symbol, bars, signals_lr)
        returns_lr = portfolio_lr.backtest_portfolio()

        bench_ret=(bars['Close'][-1]-bars['Close'][0])*100/bars['Close'][0]
        lr_ret=(returns_lr['total'][-1]-returns_lr['total'][0])*100/returns_lr['total'][0]
        svc_ret=(returns_svc['total'][-1]-returns_svc['total'][0])*100/returns_svc['total'][0]
        rf_ret=(returns_rf['total'][-1]-returns_rf['total'][0])*100/returns_rf['total'][0]


        f, ax = plt.subplots(4, sharex=True)
        f.patch.set_facecolor('white')
        ylabel = symbol + ' Close Price in Rs'
        bars['Close'].plot(ax=ax[0], color='r', lw=1.)
        ax[0].set_ylabel(ylabel, fontsize=10)
        ax[0].set_xlabel('', fontsize=14)
        ax[0].legend(('Close Price CNX-NIFTY [Return %.2f]%%'%bench_ret,), loc='upper left', prop={"size": 12})
        ax[0].set_title('CNX-NIFTY Close Price VS Portfolio Performance for Training Data ('+start_date+' to '
                        +start_test+') Test Data ('
                        +start_test+' to '+end_date+')', fontsize=14,
                        fontweight="bold")

        returns_svc['total'].plot(ax=ax[1], color='b', lw=1.)
        ax[1].set_ylabel('Portfolio value in Rs', fontsize=10)
        ax[1].set_xlabel('Date', fontsize=14)
        ax[1].legend(('Portfolio Performance.SVC [Return %.2f]%%'%svc_ret,), loc='upper left', prop={"size": 12})
        plt.tick_params(axis='both', which='major', labelsize=10)

        returns_rf['total'].plot(ax=ax[2], color='k', lw=1.)
        ax[2].set_ylabel('Portfolio value in Rs', fontsize=10)
        ax[2].set_xlabel('Date', fontsize=14)
        ax[2].legend(('Portfolio Performance.RF [Return %.2f]%%'%rf_ret,), loc='upper left', prop={"size": 12})
        plt.tick_params(axis='both', which='major', labelsize=10)

        returns_lr['total'].plot(ax=ax[3], color='g', lw=1.)
        ax[3].set_ylabel('Portfolio value in Rs', fontsize=10)
        ax[3].set_xlabel('Date', fontsize=14)
        ax[3].legend(('Portfolio Performance.LR [Return %.2f]%%'%lr_ret,), loc='upper left', prop={"size": 12})
        plt.tick_params(axis='both', which='major', labelsize=10)


        print("Benchmark Return [%.2f]%%" %(bench_ret))
        print("LR        Return [%.2f]%%" %(lr_ret))
        print("SVC       Return [%.2f]%%" %(svc_ret))
        print("RF        Return [%.2f]%%" %(rf_ret))

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        final_result +='Training Data from (' + start_date +' to '+ start_test + ') Test Data (' + start_test +' to ' + end_date + ') Benchmark Return [%.2f]%% ' %(bench_ret) +" LR Return [%.2f]%%" %(lr_ret)+" SVC Return [%.2f]%%" %(svc_ret) +" RF Return [%.2f]%%" %(rf_ret)+'\n'
        plt.show()
    print(final_result)
