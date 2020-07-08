from scipy import optimize, integrate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import time
import random
import os

from sklearn.linear_model import LinearRegression,BayesianRidge,LogisticRegression,Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns

from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network


# load data:
data_path = "./data_json/"
#text_path = "/home/dmg/CP4/news_match/news_narrative/"
with open(data_path + "twitter_time_series.json") as f:
    twitter_file = json.loads(f.read())
    twitter = {k: pd.read_json(v, orient='columns') for k, v in twitter_file.items()}

with open(data_path + "youtube_time_series.json") as f:
    youtube_file = json.loads(f.read())
    youtube = {k: pd.read_json(v, orient='columns') for k, v in youtube_file.items()}
'''
with open(text_path + "gdelt_time_series.json",encoding="utf-8") as f:
    gdelt_file = json.loads(f.read())
    gdelt = {k: pd.read_json(v, typ='series') for k, v in gdelt_file.items()}

with open(text_path + 'corrmat.json', 'r') as f:
    corr = json.loads(f.read())

eventCodeMap = corr['eventCodeMap']
narrativeMap = corr['narrativeMap']
twitterGdeltMat = np.array(corr['youtubeGdeltMat'])
twitterGdeltMat[np.array(corr['youtubeGdeltMat']) == -2] = -1
youtubeGdeltMat = np.array(corr['youtubeGdeltMat'])
youtubeGdeltMat[np.array(corr['youtubeGdeltMat']) == -2] = -1
eventCode = [event for event in eventCodeMap]
narrative = [event for event in twitter]

with open(text_path + 'corrmat.json', 'r') as f:
    corr_text = json.loads(f.read())
twitterGdeltMat_text = np.array(corr_text['twitterGdeltMat'])
narrativeMap_text = corr_text['narrativeMap']
'''
narrative = ['international/respect_sovereignty', 'maduro/dictator', 'other/chavez', 'other/chavez/anti', 'guaido/legitimate', 'military', 'violence', 'violence/against_opposition', 'international/aid', 'crisis', 'crisis/lack_essentials', 'arrests/opposition', 'arrests', 'maduro/legitimate', 'protests', 'other/chavez/pro', 'other/anti_socialism', 'maduro/narco', 'arrests/opposition/media', 'crisis/looting', 'maduro/russia_support', 'guaido/us_support', 'maduro/illegitimate', 'assembly/legitimate', 'international/military', 'arrests/opposition/protesters', 'international/aid_rejected', 'other/restore_democracy', 'other/media_bias', 'maduro/events', 'other/planned_coup', 'maduro/cuba_support', 'maduro/illegitimate/international', 'military/desertions', 'other/censorship_outage', 'international/us_sanctions', 'international/emigration', 'guaido/legitimate/international', 'violence/against_opposition/protesters', 'violence/against_maduro', 'international/break_us_relations', 'guaido/illegitimate', 'maduro/events/pro', 'maduro/legitimate/international', 'maduro/events/anti', 'other/request_observers', 'assembly/illegitimate']
def evaluation(Y_test, Y_pred):
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test) + 0.1), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / sum(Y_test)
    return rmse, ape


selected_na =["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected",
"international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate",
"maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage",
"other/chavez","other/chavez/anti","protests","violence"]

keyword_file = json.load(open("darpa_data_raw/nar_keywords.txt","r"))
#len(keyword_file)

def postprocess(pred):
    pred = np.array([int(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred
    
def dateGenerator(span):
    now = datetime.datetime(2018, 12, 24)
    delta = datetime.timedelta(days=1)
    endnow = now+datetime.timedelta(days=span)
    endnow = str(endnow.strftime('%Y-%m-%d'))
    offset = now

    Count = []
    while str(offset.strftime('%Y-%m-%d')) != endnow:
        tmp = int(time.mktime(offset.timetuple()) * 1000.0 + offset.microsecond / 1000.0)
        Count.append(str(offset.strftime('%Y-%m-%d')))
        offset += delta
    return Count

def dateGenerator_output(span):
    now = datetime.datetime(2018, 12, 24) - datetime.timedelta(hours=6, minutes=0, seconds=0)
    delta = datetime.timedelta(days=1)
    endnow = now+datetime.timedelta(days=span)
    endnow = str(endnow.strftime('%Y-%m-%d'))
    offset = now

    Count = []
    while str(offset.strftime('%Y-%m-%d')) != endnow:
        tmp = int(time.mktime(offset.timetuple()) * 1000.0 + offset.microsecond / 1000.0)
        #Count.append(str(offset.strftime('%Y-%m-%d')))
        Count.append(tmp)
        offset += delta
    return Count

def dataloader(narrative, topN, option):
    cut = 0.05
    X_train, X_test, Y_train, Y_test = [], [], [], []
    if option == "EventCount":
        Y_train = twitter[narrative].EventCount.tolist()[:25]
        Y_test = twitter[narrative].EventCount.tolist()[25:39]
    elif option == "UserCount":
        Y_train = twitter[narrative].UserCount.tolist()[:25]
        Y_test = twitter[narrative].UserCount.tolist()[25:39]
    else:
        Y_train = twitter[narrative].NewUserCount.tolist()[:25]
        Y_test = twitter[narrative].NewUserCount.tolist()[25:39]
        
    keyword = sorted([word for word in keyword_file[narrative]])[:topN]
    select_date = dateGenerator(70)
    for date in select_date[:25]:
        total = sum([keyword_file[narrative][word][date] for word in keyword])
        pop = 0
        for item in select_date[:39]:
            pop += sum([keyword_file[narrative][word][item] for word in keyword])
        if total/pop > cut:
            total = [total * 1]
        else:
            total = [total]
#         #X_train.append(np.array([keyword_file[narrative][word][date] for word in keyword]))
        X_train.append(total)
    for date in select_date[25:39]:
        total = sum([keyword_file[narrative][word][date] for word in keyword])
        pop = 0
        for item in select_date[:39]:
            pop += sum([keyword_file[narrative][word][item] for word in keyword])
        if total/pop > cut:
            total = [total * 1]
        else:
            total = [total]
        X_test.append(total)
        #X_test.append(np.array([keyword_file[narrative][word][date] for word in keyword]))
    return X_train, X_test, Y_train, Y_test, keyword

def nonlinear(X_train, Y_train, X_test):
    X = [[sum(item)] for item in X_train]
    regression = LinearRegression(fit_intercept=False,normalize=True)
    regression.fit(X, np.log(np.array(Y_train) + 1))
    for i in range(len(X_train)):
        X_train[i] = np.exp(regression.predict([[item] for item in X_train[i]]))
    for i in range(len(X_test)):
        X_test[i] = np.exp(regression.predict([[item] for item in X_test[i]]))
    return X_train, X_test

def draw(Y_train, Y_test, Y_pred):
    plt.plot(Y_train + Y_test)
    plt.plot(range(len(Y_train), len(Y_train + Y_test)),  Y_pred)
    #ratio = 1. * max(twitter[narrative[index]].EventCount.tolist()[:split]) / max(Y_val[split * index: split * (index + 1)])
    #plt.plot(range(split,40), Y_pred[14 * index: 14 * (index + 1)])
    #plt.plot(range(40), Y_train[40 * index: 40 * (index + 1)])
    plt.legend(['GT', 'Prediction'], frameon=False)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.xticks(np.arange(0, 39, 3), date[:39:3], rotation='60')
    #plt.title("Prediction for EventCount of %s" %narrative[index])
    plt.tight_layout()
    #plt.savefig("fig/%d.pdf" % index)
    plt.show()

topN = 5
blacklist = ['arrests/opposition/protesters','international/emigration', 'maduro/illegitimate/international', 'other/request_observers']
result = dict()
for item in narrative:
    result[item] = {'EventCount':[], 'UserCount':[], 'NewUserCount':[]}
#for option in ['EventCount', 'UserCount', 'NewUserCount']:
for option in ["EventCount", "UserCount", "NewUserCount"]:
    rmse, ape, size = 0, 0, 0
    for item in narrative:
        X_train, X_test, Y_train, Y_test, keyword = dataloader(item, topN, option)
        #X_tarin, X_test = nonlinear(X_train, Y_train, X_test)
#         cubic_featurizer = PolynomialFeatures(degree=1)
#         X_train_cubic = cubic_featurizer.fit_transform(X_train)
#         X_test_cubic = cubic_featurizer.transform(X_test)
#         regressor_cubic = LinearRegression()
#         regressor_cubic.fit(X_train_cubic, Y_train)
#         Y_pred = postprocess(regressor_cubic.predict(X_test_cubic))
        regression = LinearRegression(fit_intercept=False,normalize=True)
        #regression = svm.SVR()
        
        regression.fit(X_train, np.array(Y_train))
        Y_pred = postprocess(regression.predict(X_test))
        
        if item in selected_na:
            rmse_, ape_ = evaluation(Y_test, Y_pred)
            rmse += rmse_
            ape += ape_
            size += sum(Y_pred)
            plt.figure()
            plt.plot(Y_test)
            plt.plot(Y_pred)
            plt.legend(['data', 'fit'])
            plt.title("mape = {}".format(ape_))
            name = item.replace('/','#')
            fname = os.path.join("linear_model",option+"_"+name+"sum5.png")
            plt.savefig(fname, bbox_inches='tight')
        result[item][option] =  Y_pred
        index = narrative.index(item)
        #draw_m2(index, result[narrative[index]][option])
    print(rmse/len(selected_na),  ape/len(selected_na), size)
#     rmse, ape = 0, 0
#     for index in range(len(narrative)):
#         X_train, X_test, Y_train, Y_test, keyword = dataloader(narrative[index], topN, option)
#         if narrative[index] in blacklist:
#             sim, best = 0, 0
#             for i in range(len(narrative)):
#                 tmp = np.dot(twitterGdeltMat_text[:,index], twitterGdeltMat_text[:,i])
#                 if i != index and tmp > best and narrative[i] not in blacklist:
#                     sim = i
#                     best = tmp
#             result[narrative[index]][option] = result[narrative[sim]][option]
#         rmse_, ape_ = evaluation(Y_test, result[narrative[index]][option])
#         rmse += rmse_
#         ape += ape_
#         draw_m2(index, result[narrative[index]][option])
#     print("RMSE: %f, APE: %f" %(rmse/len(narrative), ape/len(narrative)))
    
for item in narrative:
    for i in range(len(result[item]['UserCount'])):
        if result[item]['UserCount'][i] < result[item]['NewUserCount'][i]:
            result[item]['UserCount'][i] = result[item]['NewUserCount'][i]
# date = dateGenerator_output(70)
# output = dict()
# for i in range(len(narrative)):
#     data = np.array([result[narrative[i]][item] for item in result[narrative[i]]]).T
#     df = pd.DataFrame(data, columns = [item for item in result[narrative[i]]], index = date[25:39])
#     output[narrative[i]] = df.to_json(orient="columns",force_ascii=False)
# with open("output/twitter_timeseries_keyword.json",'w') as wf:
#     json.dump(output, wf)
# with open("output/twitter_timeseries_keyword.json",'r') as f:
#     output_file = json.loads(f.read())
#     output = {k: pd.read_json(v, orient='columns') for k, v in output_file.items()}

#output
