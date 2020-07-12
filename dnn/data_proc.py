import pandas as pd
import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import copy

infoIDs_twitter = [u'guaido/illegitimate', u'protests', u'maduro/events/anti', u'assembly/legitimate', u'maduro/russia_support', u'international/military', u'arrests', u'international/respect_sovereignty', u'maduro/cuba_support', u'international/aid_rejected', u'violence', u'maduro/narco', u'other/planned_coup', u'crisis', u'other/restore_democracy', u'guaido/us_support', u'international/break_us_relations', u'arrests/opposition/media', u'other/anti_socialism', u'maduro/legitimate', u'international/emigration', u'crisis/looting', u'other/chavez/anti', u'maduro/dictator', u'arrests/opposition', u'other/request_observers', u'international/aid', u'maduro/illegitimate', u'maduro/events/pro', u'arrests/opposition/protesters', u'guaido/legitimate', u'crisis/lack_essentials', u'maduro/events', u'international/us_sanctions', u'maduro/illegitimate/international', u'guaido/legitimate/international', u'violence/against_opposition', u'violence/against_opposition/protesters', u'violence/against_maduro', u'maduro/legitimate/international', u'other/media_bias', u'other/chavez/pro', u'other/chavez', u'other/censorship_outage', u'military', u'military/desertions', u'assembly/illegitimate']
dict_infoID_twitter = dict()
ind = 0
for infoID in infoIDs_twitter:
	dict_infoID_twitter[infoID] = ind
	ind += 1

# print dict_infoID_twitter

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

	dd = {k: pd.read_json(v, orient='columns') for k, v in d.items()}

	index = list(dd.keys())
	column = dd[index[0]].index.values[:-1]

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

	with open('./data_json/corrmat_bert_all_norm.json', 'r') as f:
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

	df_twitter.to_csv('./data_csv/corr_bert_twitter.csv')
	df_youtube.to_csv('./data_csv/corr_bert_youtube.csv')

def generate_training_data(platform, asp, K=10):
	'''
	Generate training data from csv file
	N: for each infoID, find top N correlated eventIDs
	'''

	# read narrative time series
	df_event = pd.read_csv('./data_csv/'+platform+'_'+asp+'.csv', header=0,index_col=0)

	# read GDELT time series
	df_gdelt = pd.read_csv('./data_csv/gdelt.csv', header=0,dtype={'InfoID':str})
	df_gdelt.set_index('InfoID', inplace=True)

	# read correlation
	df_corr = pd.read_csv('./data_csv/corr_bert_'+platform+'.csv', header=0,index_col=0)

	infoIDs = sorted(df_event.index.values) # narrative
	# eventIDs = df_corr.columns.values # GDELT

	df_gdelt = df_gdelt.sort_index()
	
	# Find the popular event ID we use as the input
	active_event = df_gdelt[df_gdelt.sum(axis=1).gt(0)].index.values

	# active_gdelt = list(active_gdelt)
	df_gdelt = df_gdelt.loc[active_event,:]
	df_corr = df_corr.loc[:,active_event]
	'''
	# look for top3 corrs
	for info in infoIDs:
		threshold = df_corr.loc[info].nlargest(3)[2]
		df_corr.loc[info][df_corr.loc[info].lt(threshold)] = 0
	'''
	# filter out extreme values
	#df_corr[df_corr.lt(0.3)] = 0
    
	# k+1,K+2,...,k+n
	k=-2
	n=5
	path = './train/'+platform+'/'+str(n)
	if not os.path.exists(path):
		os.mkdir(path)

	# normalize gdelt
	arr_gdelt = df_gdelt.values # gdelta

	gdelt_norm = []
	for item in arr_gdelt:
		gdelt_norm.append(item/max(item))# normalized gdelta
	gdelt_norm = np.asarray(gdelt_norm, dtype = np.float64)

	fx_train = open(os.path.join(path, 'bert_'+asp+'_x.csv'), 'w')
	fi_train = open(os.path.join(path, 'bert_'+asp+'_ind.csv'), 'w')
	fy_train = open(os.path.join(path, 'bert_'+asp+'_y.csv'), 'w')


	ind = 0
	for infoID in infoIDs:
		BAD = ["arrests/opposition", "maduro/narco", "arrests/opposition/media", "crisis/looting", "maduro/illegitimate", "other/censorship_outage", "international/emigration", "violence/against_opposition/protesters", "international/break_us_relations", "maduro/events/pro", "maduro/legitimate/international", "maduro/events/anti", "other/request_observers", "assembly/illegitimate"]
		REALLY_BAD = ["maduro/legitimate/international", "maduro/events/anti", "other/request_observers", "assembly/illegitimate"]

		# scale gdelt
		arr_event = df_event.loc[infoID,:].values
		'''scale = max(arr_event[:26])
		arr_event = arr_event/scale'''

		w = df_corr.loc[infoID,:].values
		W = []
		for j in range(n):
			W.append(w)

		for i in range(2,25):
			I = arr_gdelt[:,i+k:i+k+n].T
			x = I*np.array(W)
			x = x.flatten()
			y = arr_event[i]

			fx_train.write(','.join(str(e) for e in x)+'\n')
			fi_train.write(str(ind)+'\n')
			fy_train.write(str(y)+'\n')


		id = infoID.replace('/','#')
		fx = open('./test/'+platform+'/'+id+'_bert_'+asp+'_x.csv', 'w')
		fi = open('./test/'+platform+'/'+id+'_bert_'+asp+'_ind.csv', 'w')
		fy = open('./test/'+platform+'/'+id+'_bert_'+asp+'_y.csv', 'w')
		fp = open('./test/'+platform+'/'+id+'_bert_'+asp+'_y_prev.csv', 'w')

		for i in range(25,39):
			I = arr_gdelt[:,i+k:i+k+n].T
			x = I*np.array(W)
			x = x.flatten()
			y = arr_event[i]
			fx.write(','.join(str(e) for e in x)+'\n')
			fi.write(str(ind)+'\n')
			fy.write(str(y)+'\n')

		ind += 1

		y_prev = arr_event[0:25]
		fp.write(','.join(str(e) for e in y_prev)+'\n')
		fx.close()
		fy.close()
		fp.close()

	fx_train.close()
	fy_train.close()


if __name__ == '__main__':

	gdelt_to_csv()
	corr_to_csv()

	platforms = ['twitter', 'youtube']
	for plf in platforms:
		json_to_csv(plf)
		aspects = ['event', 'user', 'newuser']
		for asp in aspects:
			generate_training_data(plf, asp)
	# generate_training_data('twitter')
	# narr_cluster('twitter')
