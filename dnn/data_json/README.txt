========= 1. For Twitter and YouTube Time Series Data =========

    import json

    with open('...', 'r') as f:
        d = json.loads(f.read())

Now 'd' holds the map INFO_ID -> JSON OF PD DATAFRAME
To convert this to a map of INFO_ID -> DATAFRAME, do the following:

    dd = {k: pd.read_json(v, orient='columns') for k, v in d.items()}

Now each INFO_ID corresponds to a data frame spanning from 2018-12-24 to 2019-02-01, has three columns. Note new users are calculated separately for each INFO_ID.


=============== 2. For GDELT Time Series Data ================

    with open('gdelt_time_series.json', 'r') as f:
        d = json.loads(f.read())
        
Now 'd' holds the map INFO_ID -> JSON OF PD SERIES
To convert this to a map of INFO_ID -> SERIES, do the following:

    dd = {k: pd.read_json(v, typ='series') for k, v in d.items()}
    
Now each INFO_ID corresponds to a series spanning from 2018-12-24 to 2019-04-04.


========= 3. For GDELT Event Code - Narrative Matrix =========

with open('corrmat.json', 'r') as f:
    d = json.loads(f.read())

eventCodeMap = d['eventCodeMap']
narrativeMap = d['narrativeMap']
twitterGdeltMat = np.array(corr['twitterGdeltMat'])
twitterGdeltMat[np.array(corr['twitterGdeltMat']) == -2] = np.nan
youtubeGdeltMat = np.array(corr['youtubeGdeltMat'])
youtubeGdeltMat[np.array(corr['youtubeGdeltMat']) == -2] = np.nan

By using two maps, one can reverse the indices of matrices to pairs of event code and narrative. There are NaNs because of empty series, the above code recovers this, you can deal with it later (because -2 cannot be a reasonable correlation score, so it is used to encode nan).
