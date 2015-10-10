import pandas as pd
import pandas.io.data
import seaborn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV


class DataProcessor:
    def __init__(self):
        pass

    def get_moving_average(self, data_frame, intervals):
        for interval in intervals:
            data_frame['MA_' + str(interval)] = pd.rolling_mean(data_frame['Adj Close'], interval)
        return data_frame

    def get_ewma(self, data_frame, intervals):
        for interval in intervals:
            data_frame['EMA_' + str(interval)] = pd.ewma(data_frame['Adj Close'], span=interval)
        return data_frame

    def prepare_data_for_classification(self, data_set, start_test):
        le = preprocessing.LabelEncoder()
        data_set['UpDown'] = data_set['Daily Return_nse']

        data_set.UpDown[data_set['Daily Return_nse'] >= 0] = 1
        data_set.UpDown[data_set['Daily Return_nse'] < 0] = -1

        data_set['UpDown'].fillna(1, inplace=True)
        #data_set.UpDown = le.fit(data_set.UpDown).transform(data_set.UpDown)
        features = data_set.columns[1:-1]

        X = data_set[features]
        y = data_set.UpDown

        X_train = X[X.index < start_test]
        y_train = y[y.index < start_test]

        X_test = X[X.index >= start_test]
        y_test = y[y.index >= start_test]

        return X_train, y_train, X_test, y_test

    def partition_data(self, data_set, count):
        le = preprocessing.LabelEncoder()
        data_set['UpDown'] = data_set['Daily Return_nse']

        data_set.UpDown[data_set['Daily Return_nse'] >= 0] = 1
        data_set.UpDown[data_set['Daily Return_nse'] < 0] = -1

        data_set['UpDown'].fillna(1, inplace=True)
        #data_set.UpDown = le.fit(data_set.UpDown).transform(data_set.UpDown)
        features = data_set.columns[1:-1]

        X = data_set[features]
        y = data_set.UpDown

        X_train = X[:len(data_set)-count]
        y_train = y[:len(data_set)-count]

        X_test = X[len(data_set)-count:]
        y_test = y[len(data_set)-count:]

        return X_train, y_train, X_test, y_test

    def apply_logistic_regressor(self, X_train, y_train, X_test, y_test, C=1):
        clf = LogisticRegression(C=C)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy for Logistic Classifier %s" % accuracy)
        return accuracy


    def apply_svc(self, X_train, y_train, X_test, y_test, kernel='linear', C=1):
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy for SVM Classifier %s" % accuracy)
        return accuracy

    def apply_knn(self, X_train, y_train, X_test, y_test):
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy for KNN Classifier %s" % accuracy)
        return accuracy

    def apply_random_forest(self, X_train, y_train, X_test, y_test):
        clf = RandomForestClassifier(n_estimators=5, n_jobs=-1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy for RF Classifier %s" % accuracy)
        return accuracy

    def select_best_param_svc(self, X_train, y_train, parameters):
        svr = SVC()
        clf = GridSearchCV(svr, parameters)
        clf.fit(X_train, y_train)
        print("Best Parameter SVC", clf.best_params_)
        return clf.best_params_

    def get_svc_prediction(self, X_train, y_train, x_predict, kernel='linear', C=1):
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)
        return clf.predict(x_predict)

    def get_randomforest_prediction(self, X_train, y_train, x_predict, n_estimators=5, n_jobs=-1):
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_estimators)
        clf.fit(X_train, y_train)
        return clf.predict(x_predict)

    def get_logistic_reg_prediction(self, X_train, y_train, x_predict):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf.predict(x_predict)
