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
import os, csv
import numpy as np
import matplotlib.pyplot as plt
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
#narrative = [event for event in twitter]
narrative = [event for event in youtube]

with open(text_path + 'corrmat_bert_all_norm.json', 'r') as f:
    corr_text = json.loads(f.read())
twitterGdeltMat_text = np.array(corr_text['twitterGdeltMat'])
youtuveGdeltMat_text = np.array(corr_text['youtubeGdeltMat'])
narrativeMap_text = corr_text['narrativeMap']

def dataloader_yt(gdelt_cut, corr_cut, option):
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight = [], [], [], [], [], []
    filted_gdelt = []
    for event in gdelt:
        if gdelt[event].sum() > gdelt_cut:
            filted_gdelt.append(event)
    for item in narrative:
        for i in range(split):
            if option == 'EventCount':
                Y_train.append(youtube[item].EventCount.tolist()[i])
            elif option == 'UserCount':
                Y_train.append(youtube[item].UserCount.tolist()[i])
            else:
                Y_train.append(youtube[item].NewUserCount.tolist()[i])
            X = []
            sample_weight.append(sum(youtube[item].EventCount.tolist()))
            for event in filted_gdelt:
                if youtubeGdeltMat[eventCodeMap[event],narrativeMap[item]] > corr_cut:
                    X.append(gdelt[event].tolist()[i] * youtubeGdeltMat[eventCodeMap[event],narrativeMap[item]])
                else:
                    X.append(0)
            X_train.append(np.array(X))
        for i in range(split, 39):
            if option == 'EventCount':
                Y_test.append(youtube[item].EventCount.tolist()[i])
            elif option == 'UserCount':
                Y_test.append(youtube[item].UserCount.tolist()[i])
            else:
                Y_test.append(youtube[item].NewUserCount.tolist()[i])
            X = []
            for event in filted_gdelt:
                if youtubeGdeltMat[eventCodeMap[event],narrativeMap[item]] > corr_cut:
                    X.append(gdelt[event].tolist()[i] * youtubeGdeltMat[eventCodeMap[event],narrativeMap[item]])
                else:
                    X.append(0)
            X_test.append(np.array(X))
        for i in range(40, 70):
            X = []
            for event in filted_gdelt:
                if youtubeGdeltMat[eventCodeMap[event],narrativeMap[item]] > corr_cut:
                    X.append(gdelt[event].tolist()[i] * youtubeGdeltMat[eventCodeMap[event],narrativeMap[item]])
                else:
                    X.append(0)
            X_pred.append(np.array(X))
    return X_train, X_test, X_pred, Y_train, Y_test, sample_weight

def dataloader(gdelt_cut, corr_cut, option):
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight = [], [], [], [], [], []
    filted_gdelt = []
    for event in gdelt:
        if gdelt[event].sum() > gdelt_cut:
            filted_gdelt.append(event)
    #print(len(filted_gdelt))
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
decay(gdelt, 0.90)
'''
for option in ['EventCount', 'UserCount', 'NewUserCount']:
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight = dataloader(7000,0, option)
    print("=========================X_train===============================")
    print(X_train[0])
    print(len(X_train))
    print("=========================X_test===============================")
    #print(X_test)
    print("=========================X_pred===============================")
    #print(X_pred)
    print("=========================Y_train===============================")
    print(Y_train)
    print(len(Y_train))
    print("=========================Y_test===============================")
    #print(Y_test)
    print("=========================sample_weight===============================")
    #print(sample_weight)
    print("========================================================")
    break
'''
result = dict()
depth1 = dict()
output = dict()
from models.linear_regr import linear_regr
model = linear_regr()
for item in narrative:
    result[item] = {'EventCount':[], 'UserCount':[], 'NewUserCount':[]}
    depth1[item] = {'EventCount':[], 'UserCount':[], 'NewUserCount':[]}
    output[item] = {'EventCount':{}, 'UserCount':{}, 'NewUserCount':{}}
#d = 0
for d in range(6):
#if d == 0:
    print("depth:", d)
    for option in ['EventCount', 'UserCount', 'NewUserCount']:
        X_train, X_test, X_pred, Y_train, Y_test, sample_weight = dataloader(7000,0, option)
        #print(len(X_train), len(X_test), len(Y_train), len(Y_test))
        '''
        regression = LinearRegression(fit_intercept=False,normalize=True)
        regression.fit(X_train, Y_train)
        Y_pred = regression.predict(X_test)
        Y_pred = postprocess(Y_pred)
        '''
        model_tree_1 = ModelTree(model, max_depth=d, min_samples_leaf=10, search_type="greedy", n_search_grid=10)
        model_tree_1.fit(np.array(X_train), np.array(Y_train))
        Y_pred = model_tree_1.predict(np.array(X_test))
        Y_pred = postprocess(Y_pred)
        
    
        X_train, X_test, X_pred, Y_train, Y_test, sample_weight = dataloader(7000, 0, option)

        for index in range(len(narrative)):
            regression = LinearRegression(fit_intercept=False,normalize=True)
            regression.fit(X_train[split * index: split * (index + 1)], Y_train[split * index: split * (index + 1)])
            pred = postprocess(regression.predict(X_test[14 * index: 14 * (index + 1)]))
            ratio = sum(pred) / (sum(Y_pred[14 * index: 14 * (index + 1)]) + 0.1)
            #ratio = 1
            #rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)],  ratio * np.array(Y_pred[14 * index: 14 * (index + 1)]))
            #rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], pred)
            #rmse += rmse_
            #ape += ape_
            result[narrative[index]][option] = postprocess(ratio * np.array(Y_pred[14 * index: 14 * (index + 1)]))
            #result[narrative[index]][option] = pred
            '''
            treemod = ModelTree(model, max_depth=1, min_samples_leaf=5, search_type="greedy", n_search_grid=10)
            treemod.fit(np.array(X_train[split * index: split * (index + 1)]), np.array(Y_train[split * index: split * (index + 1)]))
            pred1 = postprocess(treemod.predict(np.array(X_test[14 * index: 14 * (index + 1)])))
            ratio1 = sum(pred1) / (sum(Y_pred[14 * index: 14 * (index + 1)]) + 0.1)
            depth1[narrative[index]][option] = postprocess(ratio1 * np.array(Y_pred[14 * index: 14 * (index + 1)]))
            '''
        #print("RMSE: %f, APE: %f" %(rmse/len(narrative), ape/len(narrative)))
    
        rmse, ape, size = 0, 0, 0
        rmse1, ape1, size1 = 0,0,0
        for index in range(len(narrative)):

            if narrative[index] in selected_na:
                this = result[narrative[index]][option]
                rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], this)
                size += sum(this)
                #rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], pred)
                rmse += rmse_
                ape += ape_
                #draw_m2(index, result[narrative[index]][option])
                # write file
                
                sdate = 1547856000000
                for i in range(len(this)):
                    output[narrative[index]][str(sdate + i * 86400000)] = int(this[i])
                
                '''
                rmse_, ape_ = evaluation(Y_test[14 * index: 14 * (index + 1)], depth1[narrative[index]][option])
                size1 += sum(depth1[narrative[index]][option])
                rmse1 += rmse_
                ape1 += ape_
                '''
            
        print("RMSE: %f, APE: %f, Size: %f" %(rmse/len(selected_na), ape/len(selected_na), size))
        #print("RMSE for depth 1: %f, APE for depth 1: %f, Size for depth 1: %f" %(rmse1/len(selected_na), ape1/len(selected_na), size1))
'''
for na in selected_na:
    output[na] = pd.DataFrame(output[na]).to_json()
with open('youtube_bert_mixed.json', 'w') as outfile:
    json.dump(output, outfile)
'''
'''
for item in narrative:
    for i in range(len(result[item]['UserCount'])):
        if result[item]['UserCount'][i] < result[item]['NewUserCount'][i]:
            result[item]['UserCount'][i] = result[item]['NewUserCount'][i]

'''
