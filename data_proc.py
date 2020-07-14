import pandas as pd
import numpy as np
import json
import copy
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

infoIDs_twitter = [u'guaido/illegitimate', u'protests', u'maduro/events/anti', u'assembly/legitimate', u'maduro/russia_support', u'international/military', u'arrests', u'international/respect_sovereignty', u'maduro/cuba_support', u'international/aid_rejected', u'violence', u'maduro/narco', u'other/planned_coup', u'crisis', u'other/restore_democracy', u'guaido/us_support', u'international/break_us_relations', u'arrests/opposition/media', u'other/anti_socialism', u'maduro/legitimate', u'international/emigration', u'crisis/looting', u'other/chavez/anti', u'maduro/dictator', u'arrests/opposition', u'other/request_observers', u'international/aid', u'maduro/illegitimate', u'maduro/events/pro', u'arrests/opposition/protesters', u'guaido/legitimate', u'crisis/lack_essentials', u'maduro/events', u'international/us_sanctions', u'maduro/illegitimate/international', u'guaido/legitimate/international', u'violence/against_opposition', u'violence/against_opposition/protesters', u'violence/against_maduro', u'maduro/legitimate/international', u'other/media_bias', u'other/chavez/pro', u'other/chavez', u'other/censorship_outage', u'military', u'military/desertions', u'assembly/illegitimate']
dict_infoID_twitter = dict()
ind = 0
for infoID in infoIDs_twitter:
	dict_infoID_twitter[infoID] = ind
	ind += 1

# 
dict_infoID_twitter

def json_to_csv(platform):
	'''
	from json file to csv file
	platform: twitter or youtube
	csv file format:
	InfoID 2018-12-24, 2018-12-25, ..., 
	ID1        x           x
	'''
	with open('./data_json/'+platform+'_time_series.json', 'r') as f:
	    d = json.loads(f.read())

	# print d.keys()

	dd = {k: pd.read_json(v, orient='columns') for k, v in d.items()}

	index = list(dd.keys())
	column = dd[index[0]].index.values[:-1]

	#print(index)

	arr_event = []
	arr_user = []
	arr_newuser = []

	for key in index:
		arr_event.append(dd[key]['EventCount'].values[:-1])
		arr_user.append(dd[key]['UserCount'].values[:-1])
		arr_newuser.append(dd[key]['NewUserCount'].values[:-1])

	arr_event = np.array(arr_event)
	arr_user = np.array(arr_user)
	arr_newuser = np.array(arr_newuser)

	df_event = pd.DataFrame(arr_event, index=index, columns=column)
	df_user = pd.DataFrame(arr_user, index=index, columns=column)
	df_newuser = pd.DataFrame(arr_newuser, index=index, columns=column)

	df_event.to_csv('./data_csv/'+platform+'_event.csv', index_label='InfoID')
	df_user.to_csv('./data_csv/'+platform+'_user.csv', index_label='InfoID')
	df_newuser.to_csv('./data_csv/'+platform+'_newuser.csv', index_label='InfoID')

def gdelt_to_csv():
	# GDELT time series: form json to csv
	with open('./data_json/gdelt_time_series.json', 'r') as f:
	    d = json.loads(f.read())
	gdelt = {k: pd.read_json(v, typ='series') for k, v in d.items()}
    
	# GDELT decay
	decay_factor = 0.88
	Y = copy.deepcopy(gdelt)
	for event in gdelt:
		date = gdelt[event].index
		for i in range(len(date)):
			if i == 0:
				continue
			gdelt[event][date[i]] = decay_factor * gdelt[event][date[i-1]] + Y[event][date[i]] - Y[event][date[i-1]]
    
	index = list(gdelt.keys())
	column = gdelt[index[0]].index.values

	arr = []
	for key in index:
		arr.append(gdelt[key].values)

	arr = np.array(arr)
	df = pd.DataFrame(arr, index=index, columns=column)
	df.to_csv('./data_csv/gdelt.csv', index_label='InfoID')

def corr_to_csv():
	# correlation between infoID (narrative) and eventID (GDELT event)
	# Output csv file format:
	#          eventID1, eventID2, ...
	# infoID1,    c11  ,    c12  , ...
	# infoID2,    c21  ,    c22  , ...

	with open('./data_json/corrmat_west_en_norm.json', 'r') as f:
	    d = json.loads(f.read())

	corr_twitter = np.array(d['twitterGdeltMat'])
	corr_youtube = np.array(d['youtubeGdeltMat'])

	x = d['eventCodeMap'].items()
	x = sorted(x, key=lambda a:a[1])
	eventID = [key for key,ind in x]
	x = d['narrativeMap'].items()
	x = sorted(x, key=lambda a:a[1])
	infoID = [key for key,ind in x]

	df_twitter = pd.DataFrame(corr_twitter.T, index=infoID, columns=eventID)
	df_youtube = pd.DataFrame(corr_youtube.T, index=infoID, columns=eventID)

	df_twitter.to_csv('./data_csv/corr_west_twitter.csv')
	df_youtube.to_csv('./data_csv/corr_west_youtube.csv')

def generate_training_data(platform, section, K=10):
	'''
	Generate training data from csv file
	N: for each infoID, find top N correlated eventIDs
	'''
	path = './' + platform+ '_' + section +'/'
	# read narrative time series
	df_event = pd.read_csv('./data_csv/'+platform+'_' + section + '.csv', header=0,index_col=0)

	# read GDELT time series
	df_gdelt = pd.read_csv('./data_csv/gdelt.csv', header=0,dtype={'InfoID':str})
	df_gdelt.set_index('InfoID', inplace=True)

	# read correlation
	df_corr = pd.read_csv('./data_csv/corr_bert_'+platform+'.csv', header=0,index_col=0)

	infoIDs = sorted(df_event.index.values) # narrative
	# eventIDs = df_corr.columns.values # GDELT

	df_gdelt = df_gdelt.sort_index()

	# Find the popular event ID we use as the input
	active_event = df_gdelt[df_gdelt.sum(axis=1).gt(500)].index.values

	# active_gdelt = list(active_gdelt)
	df_gdelt = df_gdelt.loc[active_event,:]
	df_corr = df_corr.loc[:,active_event]
	#df_corr[df_corr.lt(0.3)] = 0
    
	# k+1,K+2,...,k+n
	k=0
	n=1
	#path = './train/'+platform+'/'+str(n)
	if not os.path.exists(path):
		os.mkdir(path)

	arr_gdelt = df_gdelt.values


	ind = 0
	for infoID in infoIDs:
		nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected",
					"international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate",
					"maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage",
					"other/chavez","other/chavez/anti","protests","violence"]
		if infoID not in nodelist:
			continue
		arr_event = df_event.loc[infoID,:].values
		w = df_corr.loc[infoID,:].values
		W = []
		for j in range(n):
			W.append(w)
		id = infoID.replace('/','#')
		f_train = open(os.path.join(path, id+'_bert_1day_500_decay_dryrun_train.csv'), 'w')
		for i in range(0,39):
			I = arr_gdelt[:,i+k:i+k+n].T
			x = I*np.array(W)
			x = x.flatten()
			y = arr_event[i]
			if i == 0:
				f_train.write(','.join("x"+str(cn) for cn in range(len(x))))
				f_train.write(",y\n")
			f_train.write(','.join(str(e) for e in x))
			f_train.write(','+str(y)+'\n')
		f_train.close()

		f_test = open(os.path.join(path, id+'_bert_1day_500_decay_dryrun_test.csv'), 'w')
		for i in range(39,53):
			I = arr_gdelt[:,i+k:i+k+n].T
			x = I*np.array(W)
			x = x.flatten()
			#y = arr_event[i]
			if i == 39:
				f_test.write(','.join("x"+str(cn) for cn in range(len(x))))
				f_test.write(",y\n")
			f_test.write(','.join(str(e) for e in x))
			f_test.write(','+'0'+'\n')

		ind += 1

		#y_prev = arr_event[0:25]
		f_test.close()
	assert(ind == 18)


if __name__ == '__main__':

	gdelt_to_csv()
	#corr_to_csv()

	platforms = ['twitter', 'youtube']
	sec = ['event', 'user','newuser']
	for plf in platforms:
		json_to_csv(plf)
		for s in sec:
			generate_training_data(plf,s)
            
	#generate_training_data('twitter')
	#narr_cluster('twitter')
