import pandas as pd
import numpy as np
import json
import copy
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import datetime
import time
import os
nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected","international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate","maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage","other/chavez","other/chavez/anti","protests","violence"]
#folder = "2-15 to 2-28/output/"
'''
fnames = ["twitter_UIUC_HYBRID_TEXT_avg.json","youtube_UIUC_HYBRID_TEXT_avg.json",
          "twitter_UIUC_HYBRID_TEXT_top5.json","youtube_UIUC_HYBRID_TEXT_top5.json",
          "twitter_UIUC_HYBRID_TEXT_top10.json","youtube_UIUC_HYBRID_TEXT_top10.json",
          "twitter_UIUC_NN_TEXT_avg.json","youtube_UIUC_NN_TEXT_avg.json",
          "twitter_UIUC_NN_TEXT_top5.json","youtube_UIUC_NN_TEXT_top5.json",
          "twitter_UIUC_NN_TEXT_top10.json","youtube_UIUC_NN_TEXT_top10.json",
          "twitter_UIUC_NN_GDELT_avg.json","youtube_UIUC_NN_GDELT_avg.json",
          "twitter_UIUC_NN_GDELT_top5.json","youtube_UIUC_NN_GDELT_top5.json",
          "twitter_UIUC_NN_GDELT_top10.json","youtube_UIUC_NN_GDELT_top10.json"]
'''
fnames = ["twitter_UIUC_HYBRID_TEXT_top5.json","youtube_UIUC_HYBRID_TEXT_top5.json",
          "twitter_UIUC_NN_TEXT_top5.json","youtube_UIUC_NN_TEXT_top5.json",
          "twitter_UIUC_NN_GDELT_top5.json","youtube_UIUC_NN_GDELT_top5.json"]
          
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

'''
with open("twitter_UIUC_HYBRID_TEXT_top5.json", "r") as f:
    this = json.loads(f.read())
    fdict = {k: pd.read_json(v, orient='columns') for k, v in this.items()}
    for key in fdict.keys():
        if key == "arrests":# or key == "arrests/opposition" or key == "guaido/legitimate":
            ls0 = fdict[key].EventCount
            ls1 = fdict[key].UserCount
            ls2 = fdict[key].NewUserCount
            for i in range(14):
                fdict[key].EventCount[i] = int(ls0[i] / 2)
                fdict[key].UserCount[i] = int(ls1[i] / 2)
                fdict[key].NewUserCount[i] = int(ls2[i] / 2)
        fdict[key] = fdict[key].to_json()
    #fdict = fdict.to_json()
    with open("twitter_UIUC_HYBRID_TEXT_top5.json", "w") as outfile:
        print(type(fdict))
        json.dump(fdict, outfile)

with open("twitter_UIUC_NN_GDELT_top5.json", "r") as f:
    this = json.loads(f.read())
    fdict = {k: pd.read_json(v, orient='columns') for k, v in this.items()}
    for key in fdict.keys():
        if key == "arrests":# or key == "arrests/opposition" or key == "guaido/legitimate":
            ls0 = fdict[key].EventCount
            ls1 = fdict[key].UserCount
            ls2 = fdict[key].NewUserCount
            for i in range(14):
                fdict[key].EventCount[i] = int(ls0[i] / 2)
                fdict[key].UserCount[i] = int(ls1[i] / 2)
                fdict[key].NewUserCount[i] = int(ls2[i] / 2)
        fdict[key] = fdict[key].to_json()
    #fdict = fdict.to_json()
    with open("twitter_UIUC_NN_GDELT_top5.json", "w") as outfile:
        print(type(fdict))
        json.dump(fdict, outfile)
'''
with open("twitter_UIUC_NN_TEXT_top5.json", "r") as f:
    this = json.loads(f.read())
    fdict = {k: pd.read_json(v, orient='columns') for k, v in this.items()}
    for key in fdict.keys():
        ls0 = fdict[key].EventCount
        ls1 = fdict[key].UserCount
        ls2 = fdict[key].NewUserCount
        for i in range(14):
            fdict[key].NewUserCount[i] = int(ls1[i] / 1.5)
        fdict[key] = fdict[key].to_json()
    #fdict = fdict.to_json()
    with open("twitter_UIUC_NN_TEXT_top5.json", "w") as outfile:
        print(type(fdict))
        json.dump(fdict, outfile)


for file in fnames:
    #####################################
    # make sure number is correct       #
    # make sure event > user > newuser  #
    #                                   #
    #####################################
    with open(file) as f:
        final = json.loads(f.read())
        fdict = {k: pd.read_json(v, orient='columns') for k, v in final.items()}
        correct = {}
        for key in fdict.keys():
            correct[key] = {}
            correct[key]["EventCount"] = {}
            correct[key]["UserCount"] = {}
            correct[key]["NewUserCount"] = {}
            ls0 = fdict[key].EventCount
            ls1 = fdict[key].UserCount
            ls2 = fdict[key].NewUserCount
            for i in range(14):
                correct[key]["EventCount"][ls0.index[i]] = ls0[i]
                correct[key]["UserCount"][ls1.index[i]] = ls1[i]
                correct[key]["NewUserCount"][ls2.index[i]] = ls2[i]
            for i in range(14):
                if ls0[i] < ls1[i]:
                    fdict[key]["UserCount"][i] = ls0[i]
                    correct[key]["UserCount"][ls0.index[i]] = ls0[i]
                if fdict[key].UserCount[i] < fdict[key].NewUserCount[i]:
                    correct[key]["NewUserCount"][ls2.index[i]] = fdict[key].UserCount[i]
            #if file == "youtube_UIUC_NN_GDELT_.json" and (key == 'maduro/legitimate' or key == 'other/censorship_outage'):
              #  fdict[key].UserCount[7] = 1
               # fdict[key].NewUserCount[7] = 1
                #fdict[key].EventCount[7] = 1
            correct[key] = pd.DataFrame(correct[key]).to_json()
        with open(file[:-10] + ".json", "w") as outfile:
            json.dump(correct, outfile)
    
    #####################################
    #                                   #
    # plot data to see the best result  #
    #                                   #
    #####################################
    
    
    #####################################
    #                                   #
    # plot data to see the best result  #
    #                                   #
    #####################################
    
    platform = file.split('_')[0]
    with open('./data_json/'+platform+'_time_series_to_3_21.json', 'r') as f:
        d = json.loads(f.read())
    ddraw = {k: pd.read_json(v, orient='columns') for k, v in d.items()}
    
    
    date = dateGenerator(102)
    with open(file[:-10]+".json") as f:
        output_file = json.loads(f.read())
        output = {k: pd.read_json(v, orient='columns') for k, v in output_file.items()}
        rmse, ape, size, size_gt = 0, 0, 0, 0
        for item in nodelist:
            split = 88
            pred_len = 14
            name = item.replace("/","_")
            fig, sub = plt.subplots(3,1, figsize = (10,10))
            sub[0].title.set_text(name+", Event Sum: "+str(sum(output[item].EventCount.tolist()[:14])))
            sub[0].plot(ddraw[item].EventCount.tolist())
            sub[0].plot(range(88, 102), output[item].EventCount.tolist(), c = 'm')
            #sub[0].set_xticks(np.arange(0, len(date[:split + pred_len + 5]), 3), date[:split + pred_len + 5:3], rotation='90')
            sub[1].title.set_text(name+", User Sum: "+str(sum(output[item].UserCount.tolist()[:14])))
            sub[1].plot(ddraw[item].UserCount.tolist())
            sub[1].plot(range(88, 102), output[item].UserCount.tolist(), c = 'm')
            #sub[1].set_xticks(np.arange(0, len(date[:split + pred_len + 5]), 3), date[:split + pred_len + 5:3], rotation='90')
            sub[2].title.set_text(name+", New User Sum: "+str(sum(output[item].NewUserCount.tolist()[:14])))
            sub[2].plot(ddraw[item].NewUserCount.tolist())
            sub[2].plot(range(88, 102), output[item].NewUserCount.tolist(), c = 'm')
            plt.xticks(np.arange(0, len(date[:split + pred_len + 5]), 3), date[:split + pred_len + 5:3], rotation='90')
            sub[0].grid(axis="y")
            sub[1].grid(axis="y")
            sub[2].grid(axis="y")
            plt.tight_layout()
            #plt.savefig("fig/7-21/SIR_ConditionedGDELT/%s.pdf" % name)
            plt.savefig(file[:-10]+'_'+name+'_round6.png')
    
