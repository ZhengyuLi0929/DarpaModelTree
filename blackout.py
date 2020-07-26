import pandas as pd
import numpy as np
import json
import copy
import math
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import datetime
import time
import os
nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected","international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate","maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage","other/chavez","other/chavez/anti","protests","violence"]
bk = {'arrests': 0.3019609954442556,
 'arrests/opposition': 0.29742194695991786,
 'guaido/legitimate': 0.40713972005233234,
 'international/aid': 0.4144818812145533,
 'international/aid_rejected': 0.5192915052258001,
 'international/respect_sovereignty': 0.35443879830823743,
 'maduro/cuba_support': 0.5857696867402093,
 'maduro/dictator': 0.5362395809756679,
 'maduro/legitimate': 0.188331333785405,
 'maduro/narco': 0.46898554231443956,
 'military': 0.2943264213645418,
 'military/desertions': 0.4023711821216218,
 'other/anti_socialism': 0.6847424424558123,
 'other/censorship_outage': 0.34979469537232266,
 'other/chavez': 0.28860738394866237,
 'other/chavez/anti': 0.44206222449370725,
 'protests': 0.36466472100969916,
 'violence': 0.3498565845770635
}
#folder = "2-15 to 2-28/output/"

fnames = ["twitter_UIUC_HYBRID_TEXT_avg_newData.json","twitter_UIUC_NN_TEXT_avg_newData.json",
          "twitter_UIUC_NN_GDELT_avg_newData.json",
          "twitter_UIUC_NN_TEXT_top10_newData.json","twitter_UIUC_HYBRID_TEXT_top10_newData.json",
          "twitter_UIUC_NN_GDELT_top10_newData.json",
          "youtube_UIUC_HYBRID_TEXT_avg_newData.json","youtube_UIUC_NN_TEXT_avg_newData.json",
          "youtube_UIUC_NN_GDELT_avg_newData.json",
          "youtube_UIUC_NN_TEXT_top10_newData.json", "youtube_UIUC_HYBRID_TEXT_top10_newData.json",
          "youtube_UIUC_NN_GDELT_top10_newData.json"]

#fnames = ["twitter_UIUC_NN_GDELT_avg_newData.json", "twitter_UIUC_NN_GDELT_top10_newData.json",
#          "youtube_UIUC_NN_GDELT_avg_newData.json", "youtube_UIUC_NN_GDELT_top10_newData.json"]
          
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

          
for file in fnames:
    #####################################
    # tarek blackout                    #
    #####################################
    if file == "twitter_UIUC_HYBRID_TEXT_top10_newData.json" or file == "youtube_UIUC_HYBRID_TEXT_top10_newData.json" or file == "twitter_UIUC_HYBRID_TEXT_avg_newData.json" or file == "youtube_UIUC_HYBRID_TEXT_avg_newData.json":
        final = json.loads((open(file)).read())
        fdict = {k: pd.read_json(v, orient='columns') for k, v in final.items()}
        for key in fdict.keys():
            ls0 = fdict[key].EventCount
            ls1 = fdict[key].UserCount
            ls2 = fdict[key].NewUserCount
            D = 0
            for i in range(14):
                if i == 0:
                    continue
                if i >= 7:
                    continue
                if i == 1 or i ==3:
                    D = 0
                M = 0.5 + 0.5 * (1-math.exp(-D/2))
                f_ratio = bk[key] + (1-bk[key]) * M
                fdict[key].EventCount[i] = int(ls0[i] * f_ratio)
                fdict[key].UserCount[i] = int(ls1[i] * f_ratio)
                fdict[key].NewUserCount[i] = int(ls2[i] * f_ratio)
                D += 1
            fdict[key] = fdict[key].to_json()
        with open(file[:-5]+"_blackout.json", "w") as outfile:
            json.dump(fdict, outfile)
        continue
    
    #####################################
    # blatant blackout                  #
    #####################################
    with open(file) as f:
        final = json.loads(f.read())
        fdict = {k: pd.read_json(v, orient='columns') for k, v in final.items()}
        for key in fdict.keys():
            ls0 = fdict[key].EventCount
            ls1 = fdict[key].UserCount
            ls2 = fdict[key].NewUserCount
            for i in range(14):
                if i == 0:
                    continue
                if i >= 7:
                    continue
                fdict[key].EventCount[i] = int(ls0[i] * bk[key])
                fdict[key].UserCount[i] = int(ls1[i] * bk[key])
                fdict[key].NewUserCount[i] = int(ls2[i] * bk[key])
            fdict[key] = fdict[key].to_json()
        with open(file[:-5]+"_blackout.json", "w") as outfile:
            json.dump(fdict, outfile)
    

for file in fnames:
    #####################################
    #                                   #
    # plot data to see the best result  #
    #                                   #
    #####################################
    
    platform = file.split('_')[0]
    with open('./data_json/'+platform+'_time_series_to_3_07.json', 'r') as f:
        d = json.loads(f.read())
    ddraw = {k: pd.read_json(v, orient='columns') for k, v in d.items()}
    
    
    date = dateGenerator(88)
    with open(file[:-5]+"_blackout.json") as f:
        output_file = json.loads(f.read())
        output = {k: pd.read_json(v, orient='columns') for k, v in output_file.items()}
        rmse, ape, size, size_gt = 0, 0, 0, 0
        for item in nodelist:
            split = 74
            pred_len = 14
            name = item.replace("/","_")
            plt.figure()
            plt.title(name+", Event Sum: "+str(sum(output[item].EventCount.tolist()[:14])))
            plt.plot(ddraw[item].EventCount.tolist())
            plt.plot(range(74, 88), output[item].EventCount.tolist(), c = 'm')
            plt.xticks(np.arange(0, len(date[:split + pred_len + 5]), 3), date[:split + pred_len + 5:3], rotation='90')
            plt.grid(axis="y")
            plt.tight_layout()
            #plt.savefig("fig/7-21/SIR_ConditionedGDELT/%s.pdf" % name)
            plt.savefig(file[:-5]+'_'+name+'_round4.png')

