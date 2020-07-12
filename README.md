# 1: Seperate Tree Model

## preprocessing:
1. Current data preprocessing is in data_proc.py. This deals with the correlation map and gdelt time series.

2. Preprocessing for keyword.txt is somewhere in scratch.ipynb... I may update it later.

## Run Tree
1. To generate result for 6 different depths, run python run_tests.py

2. To output the final .json result for leaderboard testing, run python run_sample.py

## structure format
1. twitter_event, twitter_user, twitter_newuser and their respective youtube folders contain different forms of processed input data

2. ouput_twitter_event contains graphs with regard to each model.

# 2: Mixed Model

## preprocessing, training and outputting all in bigmod_data_proc.py
Just run bigmod_data_proc.py to get results

# 3: DNN Model
DNN model is in a seperate directory that just called dnn.

## Preprocessing
data_proc.py under dnn folder

## Run Model + Output result
dnn.py under dnn folder
