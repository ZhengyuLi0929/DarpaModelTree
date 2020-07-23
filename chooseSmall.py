import pandas as pd
import numpy as np
import json
import copy
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

folder = "2-15 to 2-28/output/"
fnames = ["twitter_UIUC_NN_TEXT.json","youtube_UIUC_NN_TEXT.json",
          "youtube_UIUC_NN_GDELT.json","twitter_UIUC_NN_GDELT.json",
          "twitter_UIUC_HYBRID_TEXT.json", "youtube_UIUC_HYBRID_TEXT.json"]
         #,'youtube_UIUC_HYBRID_GDELT_.json','twitter_UIUC_HYBRID_GDELT_.json']
for file in fnames:
    with open(file) as f:
        final = json.loads(f.read())
        fdict = {k: pd.read_json(v, orient='columns') for k, v in final.items()}
        orig = json.loads(open(folder+file).read())
        odict = {k: pd.read_json(v, orient='columns') for k, v in orig.items()}
        for key in fdict.keys():
            ls0 = max(fdict[key].EventCount.tolist())
            ls1 = max(fdict[key].UserCount.tolist())
            ls2 = max(fdict[key].NewUserCount.tolist())
            o0 = odict[key].EventCount.tolist()[7:]
            o1 = odict[key].UserCount.tolist()[7:]
            o2 = odict[key].NewUserCount.tolist()[7:]
            os0 = max(odict[key].EventCount.tolist()[7:])
            os1 = max(odict[key].UserCount.tolist()[7:])
            os2 = max(odict[key].NewUserCount.tolist()[7:])
            if ls0 >= os0:
                for i in range(7):
                    fdict[key].EventCount[i] = o0[i]
            if ls1 >= os1:
                for i in range(7):
                    fdict[key].UserCount[i] = o1[i]
            if ls2 >= os2:
                for i in range(7):
                    fdict[key].NewUserCount[i] = o2[i]
            fdict[key] = fdict[key].to_json()
        with open(file[:-5]+"_mixed.json", "w") as outfile:
            json.dump(fdict, outfile)
