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
def preprocess_data_for_network(series, end_train_date):
    data = series.copy()
    data = data[['<DTYYYYMMDD>','<CLOSE>']]
    data = data.iloc[::-1].reset_index(drop=True)
    data['<CLOSE_yesterday>'] = data['<CLOSE>'].shift(1)
    data.drop(data.head(1).index, inplace=True)
    data['return'] = np.log(data['<CLOSE>']/data['<CLOSE_yesterday>'])
    data.drop(['<CLOSE>','<CLOSE_yesterday>'], axis=1, inplace=True)
    # Build a complete array of all dates
    data['<DTYYYYMMDD>'] = pd.to_datetime(data['<DTYYYYMMDD>'], format='%Y%m%d')
    start = data['<DTYYYYMMDD>'].iloc[0]
    end = data['<DTYYYYMMDD>'].iloc[-1]
    df = pd.DataFrame({"Date": pd.date_range(start=start, end=end)})
    differences = pd.concat([df['Date'], data['<DTYYYYMMDD>']]).drop_duplicates(keep=False)
    for i in differences:
        new_row = {'<DTYYYYMMDD>': i, 'return': None}
        data = data.append(new_row, ignore_index=True)
    data = data.sort_values(by=['<DTYYYYMMDD>'], ascending=True)
    data.rename(columns={'<DTYYYYMMDD>': 'Date'}, inplace=True)
    data = data.set_index('Date', drop=True)
    # Separation of train and test data for the network
    #   start
    train_data_for_network = data[data.index < end_train_date]
    #   end
    return train_data_for_network
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
    UUC, UDC, DUC, DDC = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))

    Return_table_all_stock = pd.DataFrame({"Date": pd.date_range('1990-01-01', datetime.today().strftime('%Y-%m-%d'))})
    Return_table_all_stock = Return_table_all_stock.set_index('Date', drop=True)
    #------------------------------------------------------------------------
    print("Start data converting... ")
    files_base_path = os.listdir(PATH_TO_DATA)
    for file_name in files_base_path:
        print(file_name)
        # read data & preprocess on data_tse
        model_file_dir = os.path.join(PATH_TO_DATA, file_name)
        data_tse = pd.read_csv(model_file_dir)
        series_i = preprocess_data_for_network(data_tse, end_date_of_training_data)
        col_name = file_name
        Return_table_all_stock[col_name] = series_i
    for ind,row in Return_table_all_stock.iterrows():
        if sum(row.notna()) < 5:
            Return_table_all_stock.drop([ind], inplace=True)
    print("Data converted. ")
    # ------------------------------------------------------------------------
    i = 0
    for column_i in Return_table_all_stock:
        j = 0
        for column_j in Return_table_all_stock:
            ups_i, downs_i, ups_j, downs_j = 0, 0, 0, 0
            uu, ud, du, dd = 0, 0, 0, 0

            for return_t in Return_table_all_stock[column_i]:
                if math.isnan(return_t) == False :
                    ups_i += Delta(return_t, +1) * abs(return_t)
                    downs_i += Delta(return_t, -1) * abs(return_t)

            for ind , row in Return_table_all_stock.iterrows():
                if math.isnan(row[column_i]) == False and math.isnan(row[column_j]) == False:
                    if row[column_i] > 0 and row[column_j] > 0:
                        uu += abs(row[column_j])
                    if row[column_i] > 0 and row[column_j] < 0:
                        ud += abs(row[column_j])
                    if row[column_i] < 0 and row[column_j] > 0:
                        du += abs(row[column_j])
                    if row[column_i] < 0 and row[column_j] < 0:
                        dd += abs(row[column_j])


            UUC[i, j] = uu / (ups_i + 0.000001)
            UDC[i, j] = ud / (ups_i + 0.000001)
            DUC[i, j] = du / (downs_i + 0.000001)
            DDC[i, j] = dd / (downs_i + 0.000001)

            j += 1
        i += 1
        print("i , ", i)

    c1 = np.concatenate((UUC, UDC), axis=1)
    c2 = np.concatenate((DUC, DDC), axis=1)
    extended_confidence = np.concatenate((c1, c2))

    return  extended_confidence
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
def preprocess_and_Separation_data_for_model(series, test_date):
    data = series.copy()
    # Add feature to data for supervised model
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
    next_day_label = np.sign(data['<CLOSE>'].diff(1).shift(-1).values)
    next_day_label[np.where(next_day_label == 0)] = 1
    data['next_day_label'] = next_day_label
    #   end
    # Build a complete array of all dates
    data['<DTYYYYMMDD>'] = pd.to_datetime(data['<DTYYYYMMDD>'], format='%Y%m%d')
    data = data.sort_values(by=['<DTYYYYMMDD>'], ascending=True)
    data = data.set_index('<DTYYYYMMDD>', drop=True)
    #   end
    # Separation of train, validation and test data for the supervised model
    end_train_date = '2019-09-22'
    start_test_date = '2020-02-22'
    train_data_for_supervised_model = data[data.index <= end_train_date]
    train_data_for_supervised_model = train_data_for_supervised_model.dropna(axis=0)
    # -----------------------------------------------------------------
    validation_data_for_supervised_model = data[data.index > end_train_date]
    validation_data_for_supervised_model = validation_data_for_supervised_model[validation_data_for_supervised_model.index < test_date]
    validation_data_for_supervised_model = validation_data_for_supervised_model.dropna(axis=0)
    # -----------------------------------------------------------------
    test_data_for_supervised_model = data[data.index > start_test_date]
    test_data_for_supervised_model = test_data_for_supervised_model.dropna(axis=0)
    # -----------------------------------------------------------------
    if test_date in test_data_for_supervised_model.index:
        return train_data_for_supervised_model, validation_data_for_supervised_model, \
               test_data_for_supervised_model.loc[test_date], 1
    else:
        return train_data_for_supervised_model, -1, -1, -1
    #   end
def window_score_validation(model,x_validation,y_validation,c):
    s = []
    for ind in x_validation.index:
        if model.predict(x_validation.loc[ind].values.reshape(1, -1)) == y_validation.loc[ind]:
            s.append(1)
        else:
            s.append(0)
    i = len(s)-1
    sum,normal = 0,0
    for a in s:
        sum += a*((1-c)**i)
        normal += (1-c)**i
        i -= 1
    return sum/normal
def Label_diccovery_and_Label_propagation_with_PageRank(threshold, Network, PATH_TO_DATA,c, test_start_date, test_end_date):
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
            if flag != -1 and len(validation) > 100:
                y_validation = validation['next_day_label']
                validation.drop(['next_day_label'], axis=1, inplace=True)
                y_test = test['next_day_label']
                test.drop(['next_day_label'], axis=0, inplace=True)
                y_vector_true = np.insert(y_vector_true, len(y_vector_true), y_test)
                for k in best_total.index:
                    if best_total.at[k, 'Market_name'] == file_name:
                        SCORE = window_score_validation(best_total.at[k, 'model'],validation, y_validation, c)
                        best_total.at[k, 'score'] = SCORE
                        best_total.at[k, 'prediction'] = best_total.at[k, 'model'].predict(test.values.reshape(1, -1))
            else:
                y_vector_true = np.insert(y_vector_true, len(y_vector_true), 0)
        # ------------------------
        sorted_model = best_total.groupby(['Market_name']).max(['score'])
        for j in sorted_model.index:
            if sorted_model.at[j, 'score'] <= threshold:
                sorted_model.drop([j], inplace=True)
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
            final_prediction = PageRank_with_personalised_vector(Network, Init, PATH_TO_DATA=PATH_TO_DATA)
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
    return result['Daily_result'].mean()
def fine_threshold( extended_confidence, PATH, c , start_date_of_test_data, end_date_of_test_data):
    #Select the best threshold on the validation data
    list1 = np.array([0.6, 0.7, 0.8, 0.9])
    list2 = np.array([0.05, 0.02, 0.01])
    score_on_validation = np.array([])
    for i in list1:
        s = Label_diccovery_and_Label_propagation_with_PageRank(i, extended_confidence, PATH, c, start_date_of_test_data, end_date_of_test_data)
        score_on_validation = np.insert(score_on_validation, len(score_on_validation), s)
    print(score_on_validation,list1[score_on_validation.argmax()])
    for i in list2:
        print(list1[score_on_validation.argmax()] - i)
        s = Label_diccovery_and_Label_propagation_with_PageRank((list1[score_on_validation.argmax()] - i), extended_confidence, PATH, c, start_date_of_test_data, end_date_of_test_data)
        score_on_validation = np.insert(score_on_validation, len(score_on_validation), s)
        list1 = np.insert(list1, len(list1), (list1[score_on_validation.argmax()] - i))
        s = Label_diccovery_and_Label_propagation_with_PageRank((list1[score_on_validation.argmax()] + i), extended_confidence, PATH, c, start_date_of_test_data, end_date_of_test_data)
        score_on_validation = np.insert(score_on_validation, len(score_on_validation), s)
        list1 = np.insert(list1, len(list1), (list1[score_on_validation.argmax()] + i))
    print("Best threshold on validation data : ",list1[score_on_validation.argmax()])
    return list1[score_on_validation.argmax()]




#---------------------------------------------------------------------------
PATH = ".\Saham\index"
end_date_of_training_data = '2019-02-19'
start_date_of_test_data = '2020-02-22'
end_date_of_test_data = '2020-07-21'


Network = Create_Network(PATH_TO_DATA=PATH, end_date_of_training_data=end_date_of_training_data)
# np.savetxt('Network_LabelNet.txt', Network)
# Network = np.loadtxt('Network_LabelNet.txt') # Network_LabelNet

# threshold = fine_threshold() # 0.68
c = 0.2
Label_diccovery_and_Label_propagation_with_PageRank(0.68, Network, PATH, c, start_date_of_test_data, end_date_of_test_data) # 0.6419

#---------------------------------------------------------------------------
