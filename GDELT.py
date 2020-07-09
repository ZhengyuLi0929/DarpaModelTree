import copy
from scipy import optimize, integrate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import time
import random
import math
from src.utils import load_csv_data
from src.ModelTree import ModelTree
from sklearn.metrics import mean_squared_error as mse


from sklearn.linear_model import LinearRegression,BayesianRidge,LogisticRegression,Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns

# load data:
with open("darpa_data_raw/cp4_nodelist.txt",'r') as wf:
    selected_na = [line.strip() for line in wf]
    
split = 25 
platform = "twitter"
data_path = "./data_json/"
text_path = "./data_json/"
with open(data_path + platform + "_time_series.json") as f:
    twitter_file = json.loads(f.read())
    twitter = {k: pd.read_json(v, orient='columns') for k, v in twitter_file.items()}

with open(data_path + "youtube_time_series.json") as f:
    youtube_file = json.loads(f.read())
    youtube = {k: pd.read_json(v, orient='columns') for k, v in youtube_file.items()}
    
with open(text_path + "gdelt_time_series.json",encoding="utf-8") as f:
    gdelt_file = json.loads(f.read())
    gdelt = {k: pd.read_json(v, typ='series') for k, v in gdelt_file.items()}

with open(text_path + 'corrmat_bert_all_norm.json', 'r') as f:
    corr = json.loads(f.read())

eventCodeMap = corr['eventCodeMap']
narrativeMap = corr['narrativeMap']
twitterGdeltMat = np.array(corr['youtubeGdeltMat'])
twitterGdeltMat[np.array(corr['youtubeGdeltMat']) == -2] = -1
youtubeGdeltMat = np.array(corr['youtubeGdeltMat'])
youtubeGdeltMat[np.array(corr['youtubeGdeltMat']) == -2] = -1
eventCode = [event for event in eventCodeMap]
narrative = [event for event in twitter]

with open(text_path + 'corrmat_bert_all_norm.json', 'r') as f:
    corr_text = json.loads(f.read())
twitterGdeltMat_text = np.array(corr_text['twitterGdeltMat'])
narrativeMap_text = corr_text['narrativeMap']

def dataloader(gdelt_cut, corr_cut, option):
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight = [], [], [], [], [], []
    filted_gdelt = []
    for event in gdelt:
        if gdelt[event].sum() > gdelt_cut:
            filted_gdelt.append(event)
    print(len(filted_gdelt))
    for item in narrative:
        for i in range(split):
            if option == 'EventCount':
                Y_train.append(twitter[item].EventCount.tolist()[i])
            elif option == 'UserCount':
                Y_train.append(twitter[item].UserCount.tolist()[i])
            else:
                Y_train.append(twitter[item].NewUserCount.tolist()[i])
            X = []
            sample_weight.append(sum(twitter[item].EventCount.tolist()))
            for event in filted_gdelt:
                if twitterGdeltMat[eventCodeMap[event],narrativeMap[item]] > corr_cut:
                    X.append(gdelt[event].tolist()[i] * twitterGdeltMat[eventCodeMap[event],narrativeMap[item]])
                else:
                    X.append(0)
            X_train.append(np.array(X))
        for i in range(split, 39):
            if option == 'EventCount':
                Y_test.append(twitter[item].EventCount.tolist()[i])
            elif option == 'UserCount':
                Y_test.append(twitter[item].UserCount.tolist()[i])
            else:
                Y_test.append(twitter[item].NewUserCount.tolist()[i])
            X = []
            for event in filted_gdelt:
                if twitterGdeltMat[eventCodeMap[event],narrativeMap[item]] > corr_cut:
                    X.append(gdelt[event].tolist()[i] * twitterGdeltMat[eventCodeMap[event],narrativeMap[item]])
                else:
                    X.append(0)
            X_test.append(np.array(X))
        for i in range(40, 70):
            X = []
            for event in filted_gdelt:
                if twitterGdeltMat[eventCodeMap[event],narrativeMap[item]] > corr_cut:
                    X.append(gdelt[event].tolist()[i] * twitterGdeltMat[eventCodeMap[event],narrativeMap[item]])
                else:
                    X.append(0)
            X_pred.append(np.array(X))
    return X_train, X_test, X_pred, Y_train, Y_test, sample_weight

def evaluation(Y_test, Y_pred):
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test) + 0.1), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / sum(Y_test)
    return rmse, ape

def draw_m2(index, pred):
    plt.plot(twitter[narrative[index]].EventCount.tolist())
    plt.plot(range(split,39), pred)
    plt.legend(['GT', 'Prediction'], frameon=False)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.xticks(np.arange(0, len(date[:40]), 3), date[:40:3], rotation='60')
    plt.title("Prediction for EventCount of %s" %narrative[index])
    plt.tight_layout()
    #plt.savefig("fig/%d.pdf" % index)
    plt.show()

def nonlinear(gdelt):
    for event in gdelt:
        date = gdelt[event].index
        for item in date:
            gdelt[event][item] = gdelt[event][item] * (1 + log(1 + gdelt[event][item]))

def decay(gdelt, decay_factor):
    Y = copy.deepcopy(gdelt)
    for event in gdelt:
        date = gdelt[event].index
        for i in range(len(date)):
            if i == 0:
                continue
            gdelt[event][date[i]] = decay_factor * gdelt[event][date[i-1]] + Y[event][date[i]] - Y[event][date[i-1]]

def postprocess(pred):
    pred = np.array([int(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred


#nonlinear(gdelt)
#decay(gdelt, 0.95)
result = dict()
for item in narrative:
    result[item] = {'EventCount':[], 'UserCount':[], 'NewUserCount':[]}
for option in ['EventCount', 'UserCount', 'NewUserCount']:
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight = dataloader(7000,0, option)
    #print(len(X_train), len(X_test), len(Y_train), len(Y_test))
    '''
    regression = LinearRegression(fit_intercept=False,normalize=True)
    regression.fit(X_train, Y_train)
    Y_pred = regression.predict(X_test)
    for i in range(len(Y_pred)):
        if Y_pred[i] < 0:
            Y_pred[i] = 0
    '''
    from models.linear_regr import linear_regr
    model_tree_1 = ModelTree(linear_regr(), max_depth=0, min_samples_leaf=10, search_type="greedy", n_search_grid=10)
    model_tree_1.fit(np.array(X_train), np.array(Y_train))
    Y_pred = model_tree_1.predict(np.array(X_test))
    Y_pred = postprocess(Y_pred)
    
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight = dataloader(7000,0, option)

    #rmse, ape = 0, 0
    for index in range(len(narrative)):
        regression = LinearRegression(fit_intercept=False,normalize=True)
        regression.fit(X_train[split * index: split * (index + 1)], Y_train[split * index: split * (index + 1)])
        pred = postprocess(regression.predict(X_test[14 * index: 14 * (index + 1)]))
        ratio = sum(pred) / (sum(Y_pred[14 * index: 14 * (index + 1)]) + 0.1)
        #rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)],  ratio * np.array(Y_pred[14 * index: 14 * (index + 1)]))
        #rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], pred)
        #rmse += rmse_
        #ape += ape_
        result[narrative[index]][option] = postprocess(ratio * np.array(Y_pred[14 * index: 14 * (index + 1)]))
        #result[narrative[index]][option] = pred
    #print("RMSE: %f, APE: %f" %(rmse/len(narrative), ape/len(narrative)))

    blacklist = ['arrests/opposition/protesters','international/emigration', 'maduro/legitimate/international', 'other/request_observers']
    rmse, ape, size = 0, 0, 0
    for index in range(len(narrative)):
        '''
        if narrative[index] in blacklist:
            sim, best = 0, 0
            for i in range(len(narrative)):
                tmp = np.dot(twitterGdeltMat_text[:,index], twitterGdeltMat_text[:,i])
                if i != index and tmp > best and narrative[i] not in blacklist:
                    sim = i
                    best = tmp
            #result[narrative[index]][option] = result[narrative[sim]][option]
            #draw_m2(index, result[narrative[index]])
            '''
        if narrative[index] in selected_na:
            rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], result[narrative[index]][option])
            size += sum(result[narrative[index]][option])
            #rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], pred)
            rmse += rmse_
            ape += ape_
            #draw_m2(index, result[narrative[index]][option])
            
    print("RMSE: %f, APE: %f, Size: %f" %(rmse/len(selected_na), ape/len(selected_na), size))

for item in narrative:
    for i in range(len(result[item]['UserCount'])):
        if result[item]['UserCount'][i] < result[item]['NewUserCount'][i]:
            result[item]['UserCount'][i] = result[item]['NewUserCount'][i]
