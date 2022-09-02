import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from sklearn.svm import SVR
import lpputils
import datetime
import math
import pred
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from scipy.sparse import csr_matrix
from sklearn import metrics
import xgboost as xgb
from collections import Counter

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

global_keys = {}
global_drivers = {}

# Priprava ciljne spremenljivke
def calculate_travel_time(y, X):
    diff = [lpputils.tsdiff(str(a), str(b)) for a, b in zip(y, X)]
    return np.array(diff)

# vrne matriko, kjer so stolpci dnevi (ponedeljek->nedelja)

def get_hours(X):
    """ Creates  attributes by splitting 1 hour into 12 bins(5 min intervals) """
    x = X.str.split(' ', expand=True)
    # split one hour into 12 bins(5 min slots)
    matrix = np.zeros(shape=(len(X), 3))
    # get departure time
    hours = [el.split(':') for el in x[1]]
    days = [list(map(int, el.split('-'))) for el in x[0]]
    days_ = []

    for i, el in enumerate(days):
        day = pd.Timestamp(datetime.datetime(el[0], el[1], el[2])).weekday()
        # print(day)
        days_.append(day)

    for i, el in enumerate(hours):
        # assign correct slot in matrix
        hours = int(el[0])
        minutes = int(el[1])
        matrix[i][0] = hours + 1
        matrix[i][1] = (hours + 1) * (days_[i] + 1) * 6
        matrix[i][2] = minutes + 1
        # matrix[i][3] = matrix[i][0] * matrix[i][1] * matrix[i][2]
        # print(hours, minutes, matrix[i][3])

    return matrix


def get_days(X):
    x = X.str.split(' ', expand=True)
    # print(x)
    matrix = np.zeros(shape=(X.size, 1))
    days = [list(map(int, el.split('-'))) for el in x[0]]
    # print(days)

    for i, el in enumerate(days):
        day = pd.Timestamp(datetime.datetime(el[0], el[1], el[2])).weekday()
        matrix[i][0] = day + 1

    return matrix


# keep only january and november, december rides
def clean_data(data):
    df = data[data['Departure time'].str.contains('-12-|-11-')]
    return df


def filter_data(data, key):
    df = data[data['Route Direction'].str.contains(key)]
    return df


# returns dictionary, key = route, value = model for this route
def count_routes(data):
    route_names = list(data['Route Direction'].unique())
    dict_of_models = {}

    for name in route_names:
        dict_of_models[name] = 0

    return dict_of_models


# Returns matrix, columns are presence of holidays or work free day
def get_holidays(X):
    y = pd.read_csv('try4.tsv', sep='\t').fillna(0)
    matrix = np.zeros(shape=(len(X), 2))

    # loop through X, check if el had holiday or free day from work, if yes then 1 else 0 in matrix
    for i, el in enumerate(X):
        el = lpputils.parsedate(el)
        day_of_year = el.timetuple().tm_yday - 1

        holiday = y['Šolske Počitnice'][day_of_year]
        free_day = y['Dela prost dan'][day_of_year]

        if holiday:
            matrix[i][0] = 1

        if free_day:
            matrix[i][1] = 1

    return matrix


def get_temperature(X):
    """ Columns represent low temp, snow, rain """
    y = pd.read_csv('try4.tsv', sep='\t').fillna(0)
    min_pad = np.min(np.array(y['pov']))
    # print(min_pad*-1)
    # matrix = np.zeros(shape=(len(X), 4))
    matrix = np.zeros(shape=(len(X), 1))

    # loop through X, check if el had temperature lower than some threshold
    for i, el in enumerate(X):
        el = lpputils.parsedate(el)
        day_of_year = el.timetuple().tm_yday - 1
        temperature = y['pov'][day_of_year]
        rain = y['padavine'][day_of_year]
        snow = str(y['sneg'][day_of_year])

        # if temperature < -2:
        #     matrix[i][0] = 1
        #
        if rain > 33:
            matrix[i][0] = 1
        #
        # if snow == 's':
        #     matrix[i][3] = 1
        #
        # if rain > 15 and temperature < 4:
        #     matrix[i][1] = 1
        # matrix[i][0] = temperature
        # matrix[i][1] = rain

    return matrix


def get_stations(X):
    """ Adds first and last station to the matrix """
    stations_first = list(X['First station'])
    stations_last = list(X['Last station'])
    global global_keys
    keys = {}
    for j, el in enumerate(set(stations_first + stations_last)):
        keys[el] = j

    global_keys = keys
    matrix = np.zeros(shape=(len(X), len(keys)))
    # print(X.size)

    for j, s in enumerate(zip(stations_first, stations_last)):
        matrix[j][keys.get(s[0])] = 1
        matrix[j][keys.get(s[1])] = 1

    return matrix


def get_drivers(data):
    drivers = list(data['Driver ID'].unique())
    matrix = np.zeros(shape=(len(data), len(drivers)))
    global global_drivers

    keys = {}
    for k, el in enumerate(drivers):
        keys[el] = k
    global_drivers = keys

    for j, s in enumerate(data['Driver ID']):
        matrix[j][keys.get(s)] = 1
    return matrix


def hyperParameterTuning(X_train, y_train):
    parameters = {"splitter": ["best", "random"],
                  "max_depth": [1, 3, 5, 7, 9, 11, 12],
                  "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
                  "max_features": ["auto", "log2", "sqrt", None],
                  "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

    dec_tree = DecisionTreeRegressor()

    gsearch = GridSearchCV(estimator=dec_tree,
                           param_grid=parameters,
                           # scoring = 'neg_mean_absolute_error', #MAE
                           scoring='neg_mean_squared_error',  # MSE
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

    gsearch.fit(X_train, y_train)

    return gsearch.best_params_


def kernels(data):
    hours = data['Departure time'].str.split(' ', expand=True)
    hours = hours[1].str.split(':', expand=True)[0]

    res = Counter(h for h in hours).most_common(4)
    res = [h[0] for h in res]
    # print(res)
    matrix = np.zeros(shape=(len(data), 1))

    for j, h in enumerate(hours):
        if h in res:
            matrix[j][0] = 1

    return matrix


def prepare_matrix(data, flag):
    """Returns matrix and travel time if training data"""
    travel_time = 0
    X = data['Departure time']

    if flag:
        travel_time = calculate_travel_time(data['Arrival time'], X)

    """
        Now we create a matrix where columns represent days of the week
        and hours of the day
        Days_of_week = col0 = monday, col1 = tuesday...
        Hours_of_day = col0 = 0, col1 = 1...
    """
    days_of_week = get_days(X)
    # print(days_of_week)

    hours_of_day = get_hours(X)
    # print(hours_of_day)

    holidays = get_holidays(X)
    # print(holidays)

    kernel = kernels(data)
    temperature = get_temperature(X)

    # print(temperature)
    # z kernel brez temp 67.13, brez kernel in temp 67.84, brez ker z temp 67.71, ker in temp skup pa 66.78, 67.49 backup
    # stations = get_stations(data)

    """ Final matrix, concatenated days and hours columns """
    final_matrix = np.concatenate((days_of_week, hours_of_day), axis=1)
    final_matrix = np.concatenate((final_matrix, holidays), axis=1)
    final_matrix = np.concatenate((final_matrix, kernel), axis=1)
    final_matrix = np.concatenate((final_matrix, temperature), axis=1)
    # final_matrix = np.concatenate((final_matrix, stations), axis=1)

    return final_matrix, travel_time


" Necessary files to run this masterpiece: train_pred.csv, lpputils.py, try4.tsv, test_pred.csv"

if __name__ == "__main__":
    data = pd.read_csv('train.csv', delimiter='\t')
    data = clean_data(data)

    test_data = pd.read_csv('test.csv', delimiter='\t')
    test_departure = test_data['Departure time']

    " Dictionary key = route, value = model for that route "
    model_dict = count_routes(data)
    test_data_routes = test_data['Route Direction'].tolist()

    fill = get_drivers(data)
    " Create dataframes for each model, based on route ",  # try to create only 1 model
    sum = 0

    for key in model_dict.keys():
        # print(key)
        get_current_data = filter_data(data, key)

        """ Not enough data to train model """
        if len(get_current_data) < 10:
            model_dict[key] = -1
            # print('less than 10', key)
            continue

        data_for_ridge, pred_for_ridge = pred.prepare_matrix(get_current_data, 1)
        data_for_ridge = csr_matrix(data_for_ridge)
        get_current_data, prediction = prepare_matrix(get_current_data, 1)

        " Create model, train_test_split "
        # this part here is ugly, but still won competition :D
        min_e_ridge = 9999
        min_e_third = 9999
        min_e_dec = 9999
        error_ridge = 0
        error_third = 0
        error_dec = 0
        final_lm_ridge = -1
        final_lm_dec = -1
        final_lm_third = -1

        kf = KFold(n_splits=10)
        kf.get_n_splits(get_current_data)

        KFold(n_splits=10, random_state=None, shuffle=False)

        for train_index, test_index in kf.split(get_current_data):
            X_train, X_test = get_current_data[train_index], get_current_data[test_index]
            y_train, y_test = prediction[train_index], prediction[test_index]

            " Train model "
            lm_dec = xgb.XGBRegressor(verbosity=0, nthread=-1, booster='gbtree', eta=0.05, max_depth=7)
            lm_ridge = xgb.XGBRFRegressor()  
          
            " Fit and save model for later use"
            lm_ridge.fit(X_train, y_train)
            lm_dec.fit(X_train, y_train)
            
            predictions_ridge = lm_ridge.predict(X_test)
            predictions_dec = lm_dec.predict(X_test)
           
            error_ridge = metrics.mean_absolute_error(y_test, predictions_ridge)
            if error_ridge < min_e_ridge:
                min_e_ridge = error_ridge
                final_lm_ridge = lm_ridge

            error_dec = metrics.mean_absolute_error(y_test, predictions_dec)
            if error_dec < min_e_dec:
                min_e_dec = error_dec
                final_lm_dec = lm_dec

        model_dict[key] = [final_lm_dec, final_lm_ridge]
        " Coef "
        # print(lm.coef_)

        " Predict "
        predictions = (final_lm_ridge.predict(X_test) + final_lm_dec.predict(X_test)) / 2
        sum += metrics.mean_absolute_error(y_test, predictions)
    
    sum /= len(model_dict.keys())
    # sum /= 1
    print('Average MAE: ' + str(sum))
    # exit(1)

    test_data, _ = prepare_matrix(test_data, 0)
   
    comp_predicts = []

    missing_values = {}
    for key in test_data_routes:
        if key not in model_dict.keys():
            missing_values[key] = -1

    # print(missing_values)
    missing_indeces = []

    for i, entry in enumerate(test_data):
        "get model"
        route = test_data_routes[i]
        """ If model doesn't exist predict with average """
        if route not in model_dict.keys() or model_dict.get(route) == -1:
            if key == 'N ŠTEPANJSKO NAS. - PODUTIK':
                comp_predicts.append(1890)
                missing_indeces.append(i)
                continue
            else:
                comp_predicts.append(1980)
                missing_indeces.append(i)
                # povprecje?, MAE z 2000 je 102.820, , 1980 je blo 150.70 za ostale,
            # za to lahko konkretno optimiziram route "N ŠTEPANJSKO NAS. - PODUTIK"
            print('missing route: ', key)
            continue
        lm_dec = model_dict.get(route)[0]
        lm_ridge = model_dict.get(route)[1]
        result = (lm_dec.predict(entry.reshape(1, -1)) + lm_ridge.predict(entry.reshape(1, -1))) / 2
        comp_predicts.append(result)

    " Plotting "
    # plt.scatter(best_model_y, best_model_pr)
    # plt.xlabel('Y Test (True Values)')
    # plt.ylabel("Predicted")
    # plt.title('MAE: ' + str(metrics.mean_absolute_error(best_model_y, best_model_pr)))
    # plt.show()

    " Create file with predictions, predict on test data "
    # f = open("common_rain.txt", "a")
    # for i, element in enumerate(test_departure):
    #     f.write(lpputils.tsadd(element, int(comp_predicts[i])))
    #     f.write('\n')
    # f.close()  # with temp
