import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lpputils
import datetime
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.sparse import csr_matrix
from sklearn import metrics


# Priprava ciljne spremenljivke
def calculate_travel_time(y, X):
    diff = [lpputils.tsdiff(str(a), str(b)) for a, b in zip(y, X)]
    return np.array(diff)


# vrne matriko, kjer so stolpci dnevi (ponedeljek->nedelja)
def get_days(X):
    x = X.str.split(' ', expand=True)
    # print(x)
    matrix = np.zeros(shape=(X.size, 7))
    days = [list(map(int, el.split('-'))) for el in x[0]]
    # print(days)

    for i, el in enumerate(days):
        day = pd.Timestamp(datetime.datetime(el[0], el[1], el[2])).weekday()
        # print(day)
        matrix[i][day] = 1

    return matrix


# keep only january and november, december rides
def clean_data(data):
    # df = data[data['Departure time'].str.contains('-01-|-12-|-11-')]
    df = data[data['Departure time'].str.contains('-12-|-11-')]

    # df_missing = data[(data['Route'] == 30) | (data['Route'] == 40) | (data['Route'] == 42)]
    # df = pd.concat([df, df_missing], ignore_index=True)

    return df


def filter_data(data, key):
    df = data[data['Route Direction'].str.contains(key)]
    # df = data[data['Route'] == key] TODO
    return df


# returns dictionary, key = route, value = model for this route
def count_routes(data):
    route_names = list(data['Route Direction'].unique())
    # route_names = list(data['Route'].unique()) TODO
    # print(route_names)
    dict_of_models = {}

    for name in route_names:
        dict_of_models[name] = 0

    return dict_of_models


def get_hours(X):
    """ Creates  attributes by splitting 1 hour into 12 bins(5 min intervals) """
    x = X.str.split(' ', expand=True)
    # split one hour into 12 bins(5 min slots)
    matrix = np.zeros(shape=(X.size, 24 * 12))
    # get departure time
    hours = [el.split(':') for el in x[1]]

    for i, el in enumerate(hours):
        # assign correct slot in matrix
        hours_index = int(el[0]) * 12
        minute_index = math.floor(int(el[1]) / 60 * 12)
        matrix[i][hours_index + minute_index] = 1

    return matrix


# Returns matrix, columns are presence of holidays or work free day
def get_holidays(X):
    y = pd.read_csv('try4.tsv', sep='\t').fillna(0)
    matrix = np.zeros(shape=(X.size, 2))

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

    matrix = np.zeros(shape=(X.size, 3))  # TODO

    # loop through X, check if el had temperature lower than some threshold
    for i, el in enumerate(X):
        el = lpputils.parsedate(el)
        day_of_year = el.timetuple().tm_yday - 1
        temperature = y['pov'][day_of_year]
        rain = y['padavine'][day_of_year]
        snow = str(y['sneg'][day_of_year])

        if temperature < -2:
            matrix[i][0] = 1

        if rain > 20:
            matrix[i][1] = 1

        # if snow == 's':
        #     matrix[i][3] = 1

        if rain > 15 and temperature < 4:
            matrix[i][2] = 1

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

    temperature = get_temperature(X)
    # print(temperature)

    """ Final matrix, concatenated days and hours columns """
    final_matrix = np.concatenate((days_of_week, hours_of_day), axis=1)
    final_matrix = np.concatenate((final_matrix, holidays), axis=1)
    final_matrix = np.concatenate((final_matrix, temperature), axis=1)

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
    # test_data_routes = test_data['Route'].tolist() TODO

    " Create dataframes for each model, based on route "
    # Grem cez vse key-e v dictu, za vsak key dobim ven vse vnose za ta route, in potem delam s temi vnosi
    # naredim model za te vnose
    # Napovedovanje: za vsak vnos v test_data uporabim drug model
    sum = 0

    for key in model_dict.keys():
        # print(key)
        get_current_data = filter_data(data, key)

        if len(get_current_data) < 10:
            # print("ERROR")
            model_dict[key] = -1  # TODO tle dj povprecje
            continue
        # get_current_test_data = filter_data(test_data, key)
        # print(get_current_data)

        " Get data "
        get_current_data, prediction = prepare_matrix(get_current_data, 1)
        # test_data, _ = prepare_matrix(test_data, 0) TODO

        get_current_data = csr_matrix(get_current_data)
        # test_data = csr_matrix(test_data) TODO
        # print(data.shape)
        # print(prediction.shape)

        " Create model, train_test_split "
        X_train, X_test, y_train, y_test = train_test_split(get_current_data, prediction, test_size=0.3, random_state=101)

        min_e = 9999
        error = 0
        alphas = np.logspace(-4, 5, 20)
        final_lm = -1
        for alpha in alphas:
            " Train model "
            # lm = LinearRegression()
            lm = Ridge(alpha=alpha)
            " Fit and save model for later use"
            lm.fit(X_train, y_train)
            predictions = lm.predict(X_test)
            error = metrics.mean_absolute_error(y_test, predictions)
            if error < min_e:
                min_e = error
                final_lm = lm

        # model_dict[key] = lm
        model_dict[key] = final_lm

        " Coef "
        # print(lm.coef_)

        " Predict "
        predictions = final_lm.predict(X_test)
        # print(metrics.mean_absolute_error(y_test, predictions))
        sum += metrics.mean_absolute_error(y_test, predictions)
        # predictions_competition = lm.predict(test_data) TODO

    " Zdej mam svoje modele za vsak route. " \
    " Zdaj se sprehodim cez vse vnose v test_data in vsak primerek vrzem v ustrezen model "

    sum /= len(model_dict.keys())
    print('Average MAE: ' + str(sum))

    test_data, _ = prepare_matrix(test_data, 0)
    test_data = csr_matrix(test_data)

    comp_predicts = []

    missing_values = {}
    for key in test_data_routes:
        if key not in model_dict.keys():
            missing_values[key] = 0

    # print(missing_values)
    missing_indeces = []

    for i, entry in enumerate(test_data):
        "get model"
        route = test_data_routes[i]
        if route not in model_dict.keys() or model_dict.get(route) == -1:
            comp_predicts.append(2065) #povprecje?, MAE z 2000 je 102.820
            missing_indeces.append(i)
            # print('ERROR')
            # print(route)
            continue
        lm = model_dict.get(route)
        # print(lm)
        result = lm.predict(entry)
        comp_predicts.append(result)

    # predictions_competition = lm.predict(test_data)

    " Plotting "
    # plt.scatter(best_model_y, best_model_pr)
    # plt.xlabel('Y Test (True Values)')
    # plt.ylabel("Predicted")
    # plt.title('MAE: ' + str(metrics.mean_absolute_error(best_model_y, best_model_pr)))
    # plt.show()

    # print(missing_indeces)
    # print(comp_predicts)

    " Create file with predictions, predict on test data "

    # print(test_departure[0])
    # print(predictions_competition[0])
    # print(lpputils.tsadd(test_departure[0], predictions_competition[0]))
    #
    # f = open("last_try.txt", "a")
    # for i, element in enumerate(test_departure):
    #     f.write(lpputils.tsadd(element, int(comp_predicts[i])))
    #     f.write('\n')
    # f.close()

    # print(model_dict)
    # print(len(comp_predicts))
    # print(comp_predicts[2][0]) TODO ROUTE ALI ROUTE DIRECTION