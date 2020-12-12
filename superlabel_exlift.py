
import os
import math
import numpy as np
import pandas as pd
import talib
from fast_pagerank import pagerank_power
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from datetime import datetime


def preprocess_and_Separation_data_for_network(data, end_date_of_training_data):
    data = data.drop(['<TICKER>', '<VOL>', '<OPENINT>', '<PER>', '<FIRST>', '<LAST>', '<VALUE>','<LOW>','<OPEN>','<HIGH>'], axis=1)
    data = data.iloc[::-1].reset_index(drop=True)
    data['<CLOSE_yesterday>'] = data['<CLOSE>'].shift(1)
    data.drop(data.head(1).index, inplace=True)
    data['return'] = np.log(data['<CLOSE>']/data['<CLOSE_yesterday>'])
    data.drop(['<CLOSE>','<CLOSE_yesterday>'], axis=1, inplace=True)
    # Build a complete array of all dates
    #   start
    data['<DTYYYYMMDD>'] = pd.to_datetime(data['<DTYYYYMMDD>'], format='%Y%m%d')
    start = data['<DTYYYYMMDD>'].iloc[0]
    end = data['<DTYYYYMMDD>'].iloc[-1]
    df = pd.DataFrame({"Date": pd.date_range(start=start, end=end)})
    differences = pd.concat([df['Date'], data['<DTYYYYMMDD>']]).drop_duplicates(keep=False)
    del df
    for i in differences:
        new_row = {'<DTYYYYMMDD>': i, 'return': None}
        data = data.append(new_row, ignore_index=True)
    data = data.sort_values(by=['<DTYYYYMMDD>'], ascending=True)
    data.rename(columns={'<DTYYYYMMDD>': 'Date'}, inplace=True)
    data = data.set_index('Date', drop=True)
    #   end
    # Separation of train and test data for the network
    #   start
    train_data_for_network = data[:end_date_of_training_data]
    #   end
    return train_data_for_network
def preprocess_and_Separation_data_for_model(series, test_date):
    end_train_date = '2019-09-22'
    star_validation_date = '2019-09-23'
    end_validation_date = '2020-02-19'
    start_test_date = '2020-02-22'
    data = series.copy()
    # Add feature to data for supervised model
    #   start
    data = data.drop(['<TICKER>', '<VOL>', '<OPENINT>', '<PER>', '<FIRST>', '<LAST>', '<VALUE>'], axis=1)
    data = data.iloc[::-1].reset_index(drop=True)
    data['<RSI>'] = talib.RSI(data['<CLOSE>'], timeperiod=14)
    data['SMA_5'] = talib.SMA(data['<CLOSE>'], timeperiod=5)
    data['WMA_5'] = talib.WMA(data['<CLOSE>'], timeperiod=5)
    data['WMA_10'] = talib.WMA(data['<CLOSE>'], timeperiod=10)
    data['MOM_10'] = talib.MOM(data['<CLOSE>'], timeperiod=10)
    data.drop(data.head(15).index, inplace=True)
    #   end
    # Add Next Day label
    #   start
    next_day_label = np.sign(data['<CLOSE>'].diff(1).shift(-1).values)
    next_day_label[np.where(next_day_label == 0)] = 1
    data['next_day_label'] = next_day_label
    #   end
    # Build a complete array of all dates
    #   start
    data['<DTYYYYMMDD>'] = pd.to_datetime(data['<DTYYYYMMDD>'], format='%Y%m%d')
    data = data.sort_values(by=['<DTYYYYMMDD>'], ascending=True)
    data = data.set_index('<DTYYYYMMDD>', drop=True)
    #   end
    # Separation of train, validation and test data for the supervised model
    #   start
    train_data_for_supervised_model = data[:end_train_date]
    train_data_for_supervised_model = train_data_for_supervised_model.dropna(axis=0)
    # -----------------------------------------------------------------
    validation_data_for_supervised_model = data[star_validation_date:test_date]
    validation_data_for_supervised_model.drop(validation_data_for_supervised_model.tail(1).index, inplace=True)
    validation_data_for_supervised_model = validation_data_for_supervised_model.dropna(axis=0)
    test_data_for_supervised_model = data[start_test_date:]
    last_day = test_data_for_supervised_model.tail(1)
    test_data_for_supervised_model = test_data_for_supervised_model.dropna(axis=0)
    test_data_for_supervised_model = test_data_for_supervised_model.append(last_day)
    if test_date in test_data_for_supervised_model.index:
        return train_data_for_supervised_model, validation_data_for_supervised_model, \
               test_data_for_supervised_model.loc[test_date], 1
    else:
        return train_data_for_supervised_model, -1, -1, -1
    #   end
def Delta(return_t, direction):
    if return_t * direction > 0:
        return 1
    else:
        return 0
def sign(number):
    if number >= 0:
        return +1
    else:
        return -1
def Create_Network(PATH_TO_DATA, end_date_of_training_data):
    n = len([f for f in os.listdir(PATH_TO_DATA) if os.path.isfile(os.path.join(PATH_TO_DATA, f))])
    UUL, UDL, DUL, DDL = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))

    Return_table_all_stock = pd.DataFrame({"Date": pd.date_range('1990-01-01', datetime.today().strftime('%Y-%m-%d'))})
    Return_table_all_stock = Return_table_all_stock.set_index('Date', drop=True)
    #------------------------------------------------------------------------
    files_base_path = os.listdir(PATH_TO_DATA)
    for file_name in files_base_path:
        print(file_name)
        # read data & preprocess on data_tse
        model_file_dir = os.path.join(PATH_TO_DATA, file_name)
        data_tse = pd.read_csv(model_file_dir)
        series_i = preprocess_and_Separation_data_for_network(data_tse, end_date_of_training_data)
        col_name = file_name.replace(".csv", "")
        Return_table_all_stock[col_name] = series_i
    for ind,row in Return_table_all_stock.iterrows():
        if sum(row.notna()) < len(Return_table_all_stock.columns)/2:
            Return_table_all_stock.drop([ind], inplace=True)
    # ------------------------------------------------------------------------

    i = 0
    for column_i in Return_table_all_stock:
        j = 0
        for column_j in Return_table_all_stock:
            ups_i = 0
            downs_i = 0
            ups_j = 0
            downs_j = 0
            for return_t in Return_table_all_stock[column_i]:
                if math.isnan(return_t) == False :
                    ups_i += Delta(return_t, +1) * abs(return_t)
                    downs_i += Delta(return_t, -1) * abs(return_t)
            for return_t in Return_table_all_stock[column_j]:
                if math.isnan(return_t) == False :
                    ups_j += Delta(return_t, +1) * abs(return_t)
                    downs_j += Delta(return_t, -1) * abs(return_t)

            uu, ud, du, dd = 0, 0, 0, 0
            if len(Return_table_all_stock[column_i]) > len(Return_table_all_stock[column_j]):
                series_i = Return_table_all_stock[column_i].copy()
                series_j = Return_table_all_stock[column_j].copy()
                series_j = series_j.shift(-1)
                print(series_i)
                print("-------------")
                print(series_j)
                for ind in series_j.index:
                    if series_i.loc[ind] != None and series_j.loc[ind] != None :
                        if series_i.loc[ind] > 0 and series_j.loc[ind] > 0:
                            uu += abs(series_j.loc[ind])
                        if series_i.loc[ind] > 0 and series_j.loc[ind] < 0:
                            ud += abs(series_j.loc[ind])
                        if series_i.loc[ind] < 0 and series_j.loc[ind] > 0:
                            du += abs(series_j.loc[ind])
                        if series_i.loc[ind] < 0 and series_j.loc[ind] < 0:
                            dd += abs(series_j.loc[ind])
            else:
                series_i = Return_table_all_stock[column_i].copy()
                series_j = Return_table_all_stock[column_j].copy()
                series_j = series_j.shift(-1)
                for ind in series_i.index:
                    if series_i.loc[ind] != None and series_j.loc[ind] != None :
                        if series_i.loc[ind] > 0 and series_j.loc[ind] > 0:
                            uu += abs(series_j.loc[ind])
                        if series_i.loc[ind] > 0 and series_j.loc[ind] < 0:
                            ud += abs(series_j.loc[ind])
                        if series_i.loc[ind] < 0 and series_j.loc[ind] > 0:
                            du += abs(series_j.loc[ind])
                        if series_i.loc[ind] < 0 and series_j.loc[ind] < 0:
                            dd += abs(series_j.loc[ind])


            UUL[i, j] = uu / (ups_i * ups_j + 0.000001)
            UDL[i, j] = ud / (ups_i * downs_j + 0.000001)
            DUL[i, j] = du / (downs_i * ups_j + 0.000001)
            DDL[i, j] = dd / (downs_i * downs_j + 0.000001)


            j += 1

        i += 1
        print("i , ", i)
    a1 = np.concatenate((UUL, UDL), axis=1)
    a2 = np.concatenate((DUL, DDL), axis=1)
    extended_lift = np.concatenate((a1, a2))

    return extended_lift
def PageRank_with_personalised_vector(Network, Init, PATH_TO_DATA):
    n = len([f for f in os.listdir(PATH_TO_DATA) if os.path.isfile(os.path.join(PATH_TO_DATA, f))])
    pagerank = pagerank_power(Network, p=0.85, personalize=Init)
    R = np.array([])
    for v in range(0, n):
        if pagerank[v] - pagerank[v + n] > 0:
            R = np.insert(R, len(R), +1)
        else:
            R = np.insert(R, len(R), -1)
    return R
def SuperLabel_and_label_propagation_with_PageRank(threshold, w1, w2, w3, Network, PATH_TO_DATA, test_start_date, test_end_date):
    # -----------------------------------------------------------------------------------------------------------------
    # models is an array containing all the trained models for all markets
    # -----------------------------------------------------------
    models = pd.DataFrame(columns=['Market_name', 'score', 'model', 'prediction'])
    files_base_path = os.listdir(PATH_TO_DATA)
    for file_name in files_base_path:
        # read data & preprocess on data_tse
        model_file_dir_i = os.path.join(PATH_TO_DATA, file_name)
        data_tse = pd.read_csv(model_file_dir_i)
        if len(data_tse) > 300:
            train, validation, test, flag = preprocess_and_Separation_data_for_model(data_tse, '2020-02-19')
            y_train = train['next_day_label']
            train.drop(['next_day_label'], axis=1, inplace=True)
            # -----------------------------------------------------------
            LDA = LinearDiscriminantAnalysis()
            LDA.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': LDA, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            RF = RandomForestClassifier()
            RF.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': RF, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            DT = DecisionTreeClassifier()
            DT.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': DT, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            NB = GaussianNB()
            NB.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': NB, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            SVM = svm.NuSVC(gamma='auto')
            SVM.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': SVM, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            FDA = QuadraticDiscriminantAnalysis()
            FDA.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': FDA, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # ---------------------------------------------------------
            MLP = MLPClassifier()
            MLP.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': MLP, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # ---------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    # Testing method
    # Build the result array with all possible days, without Thursdays and Fridays . Attention : still includes holidays
    # -----------------------------------------------------------
    result = pd.DataFrame({"Date": pd.date_range(start=test_start_date, end=test_end_date)})
    result["Dayofweek"] = result.Date.dt.dayofweek
    for i in result.index:
        if result.at[i, 'Dayofweek'] == 3 or result.at[i, 'Dayofweek'] == 4:
            result.drop([i], inplace=True)
    result.drop(['Dayofweek'], axis=1, inplace=True)
    result['Daily_result'] = None
    result = result.set_index("Date", drop=True)
    print("Test start date , Test end date")
    print(test_start_date, test_end_date)

    # -----------------------------------------------------------
    # For each day of test data
    for i in result.index:
        print(i)
        Init_up = np.array([])
        Init_down = np.array([])
        y_vector_true = np.array([])
        # ------------------------------------------------------------------------------------------------------
        files_base_path = os.listdir(PATH_TO_DATA)
        best_total = models.copy()
        for file_name in files_base_path:
            # read data & preprocess on data_tse
            model_file_dir = os.path.join(PATH_TO_DATA, file_name)
            data_tse = pd.read_csv(model_file_dir)
            train, validation, test, flag = preprocess_and_Separation_data_for_model(data_tse, i)
            if flag != -1 and len(validation) > 90:
                y_validation = validation['next_day_label']
                validation.drop(['next_day_label'], axis=1, inplace=True)
                y_test = test['next_day_label']
                test.drop(['next_day_label'], axis=0, inplace=True)
                y_vector_true = np.insert(y_vector_true, len(y_vector_true), y_test)
                for k in best_total.index:
                    if best_total.at[k, 'Market_name'] == file_name:
                        best_total.at[k, 'model'].score(validation, y_validation)
                        SCORE_100 = best_total.at[k, 'model'].score(validation, y_validation)
                        SCORE_50 = best_total.at[k, 'model'].score(validation[len(validation) - 50:],
                                                                   y_validation[len(y_validation) - 50:])
                        SCORE_10 = best_total.at[k, 'model'].score(validation[len(validation) - 10:],
                                                                   y_validation[len(y_validation) - 10:])
                        SCORE = (w1 * SCORE_10) + (w2 * SCORE_50) + (w3 * SCORE_100)
                        best_total.at[k, 'score'] = SCORE
                        best_total.at[k, 'prediction'] = best_total.at[k, 'model'].predict(test.values.reshape(1, -1))
            else:
                y_vector_true = np.insert(y_vector_true, len(y_vector_true), 0)
        # ------------------------
        sorted_model = best_total.groupby(['Market_name']).max()
        tole = len(sorted_model)
        for j in sorted_model.index:
            if sorted_model.at[j, 'score'] <= threshold:
                sorted_model.drop([j], inplace=True)
        print("darsad node label dar :", len(sorted_model) / tole)
        # ------------------------------------------------------------------------------------------------------
        for file_name in files_base_path:

            if file_name in sorted_model.index:
                t = sorted_model.at[file_name, 'prediction']
            else:
                t = 0
            # -----------------------------------------------------------
            if t > 0:
                Init_up = np.insert(Init_up, len(Init_up), 1)
                Init_down = np.insert(Init_down, len(Init_down), 0)
            if t < 0:
                Init_up = np.insert(Init_up, len(Init_up), 0)
                Init_down = np.insert(Init_down, len(Init_down), 1)
            if t == 0:
                Init_up = np.insert(Init_up, len(Init_up), 0)
                Init_down = np.insert(Init_down, len(Init_down), 0)
        # ------------------------------------------------------------------------------------------------------
        if np.sum(np.abs(y_vector_true)) > 10:  # roze tatil nabashad
            Init = np.concatenate((Init_up, Init_down))
            final_prediction = PageRank_with_personalised_vector(Network, Init, PATH_TO_DATA=PATH)
            flag, sum = 0, 0
            for m in final_prediction * y_vector_true:
                if m != 0:
                    sum += 1
                if m == 1:
                    flag += 1
            result.at[i, 'Daily_result'] = (flag / sum)
        # ------------------------------------------------------------------------------------------------------
    print(result)
    print("*************  result Network == ", result['Daily_result'].mean(), "********************")
    return
def SuperLabel_for_tomorrow(threshold, w1, w2, w3, Network, PATH_TO_DATA, test_start_date, test_end_date):
    # -----------------------------------------------------------------------------------------------------------------
    # models is an array containing all the trained models for all markets
    # -----------------------------------------------------------
    models = pd.DataFrame(columns=['Market_name', 'score', 'model', 'prediction'])
    files_base_path = os.listdir(PATH_TO_DATA)
    for file_name in files_base_path:
        # read data & preprocess on data_tse
        model_file_dir_i = os.path.join(PATH_TO_DATA, file_name)
        data_tse = pd.read_csv(model_file_dir_i)
        if len(data_tse) > 300:
            train, validation, test, flag = preprocess_and_Separation_data_for_model(data_tse, '2020-02-19')
            y_train = train['next_day_label']
            train.drop(['next_day_label'], axis=1, inplace=True)
            # -----------------------------------------------------------
            LDA = LinearDiscriminantAnalysis()
            LDA.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': LDA, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            RF = RandomForestClassifier()
            RF.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': RF, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            DT = DecisionTreeClassifier()
            DT.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': DT, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            NB = GaussianNB()
            NB.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': NB, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            SVM = svm.NuSVC(gamma='auto')
            SVM.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': SVM, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # -----------------------------------------------------------
            FDA = QuadraticDiscriminantAnalysis()
            FDA.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': FDA, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # ---------------------------------------------------------
            MLP = MLPClassifier()
            MLP.fit(train, y_train)
            dic = {'Market_name': file_name, 'score': 0, 'model': MLP, 'prediction': 0}
            models = models.append(dic, ignore_index=True)
            # ---------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    # Testing method
    # Build the result array with all possible days, without Thursdays and Fridays . Attention : still includes holidays
    # -----------------------------------------------------------
    result = pd.DataFrame({"Date": pd.date_range(start=test_start_date, end=test_end_date)})
    result["Dayofweek"] = result.Date.dt.dayofweek
    for i in result.index:
        if result.at[i, 'Dayofweek'] == 3 or result.at[i, 'Dayofweek'] == 4:
            result.drop([i], inplace=True)
    result.drop(['Dayofweek'], axis=1, inplace=True)
    result['Daily_result'] = None
    result = result.set_index("Date", drop=True)
    print("Test start date , Test end date")
    print(test_start_date, test_end_date)

    # -----------------------------------------------------------
    # For each day of test data
    for i in result.index:
        print(i)
        Init_up = np.array([])
        Init_down = np.array([])
        y_vector_true = np.array([])
        # ------------------------------------------------------------------------------------------------------
        files_base_path = os.listdir(PATH_TO_DATA)
        best_total = models.copy()
        for file_name in files_base_path:
            # read data & preprocess on data_tse
            model_file_dir = os.path.join(PATH_TO_DATA, file_name)
            data_tse = pd.read_csv(model_file_dir)
            train, validation, test, flag = preprocess_and_Separation_data_for_model(data_tse, i)
            if flag != -1 and len(validation) > 90:
                y_validation = validation['next_day_label']
                validation.drop(['next_day_label'], axis=1, inplace=True)
                y_test = test['next_day_label']
                test.drop(['next_day_label'], axis=0, inplace=True)
                y_vector_true = np.insert(y_vector_true, len(y_vector_true), y_test)
                for k in best_total.index:
                    if best_total.at[k, 'Market_name'] == file_name:
                        best_total.at[k, 'model'].score(validation, y_validation)
                        SCORE_100 = best_total.at[k, 'model'].score(validation, y_validation)
                        SCORE_50 = best_total.at[k, 'model'].score(validation[len(validation) - 50:],
                                                                   y_validation[len(y_validation) - 50:])
                        SCORE_10 = best_total.at[k, 'model'].score(validation[len(validation) - 10:],
                                                                   y_validation[len(y_validation) - 10:])
                        SCORE = (w1 * SCORE_10) + (w2 * SCORE_50) + (w3 * SCORE_100)
                        best_total.at[k, 'score'] = SCORE
                        best_total.at[k, 'prediction'] = best_total.at[k, 'model'].predict(test.values.reshape(1, -1))
            else:
                y_vector_true = np.insert(y_vector_true, len(y_vector_true), 0)
        # ------------------------
        sorted_model = best_total.groupby(['Market_name']).max()
        tole = len(sorted_model)
        for j in sorted_model.index:
            if sorted_model.at[j, 'score'] <= threshold:
                sorted_model.drop([j], inplace=True)
        print("darsad node label dar :", len(sorted_model) / tole)
        # ------------------------------------------------------------------------------------------------------
        for file_name in files_base_path:
            if file_name in sorted_model.index:
                t = sorted_model.at[file_name, 'prediction']
            else:
                t = 0
            # -----------------------------------------------------------
            if t > 0:
                Init_up = np.insert(Init_up, len(Init_up), 1)
                Init_down = np.insert(Init_down, len(Init_down), 0)
            if t < 0:
                Init_up = np.insert(Init_up, len(Init_up), 0)
                Init_down = np.insert(Init_down, len(Init_down), 1)
            if t == 0:
                Init_up = np.insert(Init_up, len(Init_up), 0)
                Init_down = np.insert(Init_down, len(Init_down), 0)
        # ------------------------------------------------------------------------------------------------------
        Init = np.concatenate((Init_up, Init_down))
        final_prediction = PageRank_with_personalised_vector(Network, Init, PATH_TO_DATA=PATH)
        # ------------------------------------------------------------------------------------------------------

    return final_prediction


#---------------------------------------------------------------------------
PATH = ".\Saham\index"
end_date_of_training_data = '2019-02-19'
start_date_of_Validation_data = '2019-09-23'
start_date_of_test_data = '2020-07-21'
end_date_of_test_data = '2020-07-21'
tomorrow = '2020-07-22'


Network = Create_Network(PATH_TO_DATA=PATH, end_date_of_training_data=end_date_of_training_data)
# np.savetxt('Network_SuperLabel.txt', Network)
# Network = np.loadtxt('Network_SuperLabel.txt')



SuperLabel_and_label_propagation_with_PageRank(0.6, 3/6, 2/6, 1/6, Network, PATH, start_date_of_test_data,end_date_of_test_data)
prediction_tomorrow = SuperLabel_for_tomorrow(0.6, 3/6, 2/6, 1/6, Network, PATH, tomorrow,tomorrow)
print(prediction_tomorrow)

#---------------------------------------------------------------------------