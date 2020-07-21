import os, csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_csv_data
from src.ModelTree import ModelTree
import json
from sklearn.metrics import mean_squared_error as mse
from random import randrange, uniform

def main():
    nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected","international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate","maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage","other/chavez","other/chavez/anti","protests","violence"]
    targets = ["twitter_event","twitter_user","twitter_newuser","youtube_event","youtube_user","youtube_newuser"]
    
    yt = {}
    tt = {}
    error = np.zeros(2)
    for key in nodelist:
        tt[key] = {}
        yt[key] = {}
        for target in targets:
            name = key.replace('/','#')
            fname = name + "_lasso_214_train.csv"
            data_csv_data_filename = os.path.join(target,fname)
            X, y, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            # Train different depth model tree fits and plot results
            #from models.mean_regr import mean_regr
            #plot_model_tree_fit(mean_regr(), X, y, name, mapes, rmses, target)
            from models.lasso import lasso
            #from models.linear_regr import linear_regr
            #data = plot_model_tree_fit(linear_regr(), X, y, name, target, error)
            data = plot_model_tree_fit(lasso(), X, y, name, target, error)
            sdate = 1550188800000# 2-14 #1549584000000
            if target == "twitter_event":
                tt[key]["EventCount"] = {}
                for i in range(len(data)):
                    tt[key]["EventCount"][str(sdate + i * 86400000)] = int(data[i])
            if target == "twitter_user":
                tt[key]["UserCount"] = {}
                for i in range(len(data)):
                    tt[key]["UserCount"][str(sdate + i * 86400000)] = int(data[i])
            if target == "twitter_newuser":
                tt[key]["NewUserCount"] = {}
                for i in range(len(data)):
                    tt[key]["NewUserCount"][str(sdate + i * 86400000)] = int(data[i])
            if target == "youtube_user":
                yt[key]["UserCount"] = {}
                for i in range(len(data)):
                    yt[key]["UserCount"][str(sdate + i * 86400000)] = int(data[i])
            if target == "youtube_newuser":
                yt[key]["NewUserCount"] = {}
                for i in range(len(data)):
                    yt[key]["NewUserCount"][str(sdate + i * 86400000)] = int(data[i])
            if target == "youtube_event":
                yt[key]["EventCount"] = {}
                for i in range(len(data)):
                    yt[key]["EventCount"][str(sdate + i * 86400000)] = int(data[i])
        tt[key] = pd.DataFrame(tt[key]).to_json()
        yt[key] = pd.DataFrame(yt[key]).to_json()
    #error = error/108
    #print("rmse:",  error[0])
    #print("ape:", error[1])
    with open('youtube_UIUC_NN_TEXT_lasso.json', 'w') as outfile:
        json.dump(yt, outfile)
    with open('twitter_UIUC_NN_TEXT_lasso.json', 'w') as outfile:
        json.dump(tt, outfile)
# ********************************
#
# Side functions
#
# ********************************
def evaluation(Y_test, Y_pred):
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test) + 0.1), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / sum(Y_test)
    return rmse, ape
    
def postprocess(pred):
    pred = np.array([int(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred

def plot_model_tree_fit(model, X, y, name,  target, error):
        #output_filename = os.path.join("output_"+target, "west_1day_5000_linear_{}_greedy_leaf_5_{}_fit.png".format(model.__class__.__name__, name))
        #print("Saving model tree predictions plot y vs x to '{}'...".format(output_filename))

        #mape_ls = np.zeros(12)
        # random forest
        bag = 10
        placeholder = []
        for i in range(bag):
            X_real = copy.deepcopy(X)
            Y_real = copy.deepcopy(y)
            depth = randrange(4)
            leaf = randrange(5,8)
            mask_num = randrange(1,4)
            for j in range(mask_num):
                mask = randrange(len(X_real))
                X_real = np.delete(X_real, mask, 0)
                Y_real = np.delete(Y_real, mask, 0)
        #depth = 1
        #if depth == 1:
            # Form model tree
            #print(" -> training model tree depth={}...".format(depth))
            model_tree = ModelTree(model, max_depth=depth, min_samples_leaf=leaf,
                                   search_type="greedy", n_search_grid=10)
            # Train model tree
            model_tree.fit(X_real, Y_real, verbose=False)
            
            data_csv_data_filename = os.path.join(target, name+"_lasso_214_test.csv")
            X_test, y_test, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            y_train_pred = model_tree.predict (X)
            y_pred = model_tree.predict(X_test)
            y_pred = postprocess(y_pred)
            placeholder.append(y_pred)
        placeholder = np.array(placeholder)
        final_pred = placeholder.mean(axis = 0)
        #rmse, ape = evaluation(y_test, final_pred)
        #error[0] += rmse
        #error[1] += ape
        return final_pred


def generate_csv_data(func, output_csv_filename, x_range=(0, 1), N=500):
    x_vec = np.linspace(x_range[0], x_range[1], N)
    y_vec = np.vectorize(func)(x_vec)
    with open(output_csv_filename, "w") as f:
        writer = csv.writer(f)
        field_names = ["x1", "y"]
        writer.writerow(field_names)
        for (x, y) in zip(x_vec, y_vec):
            field_values = [x, y]
            writer.writerow(field_values)

# Driver
if __name__ == "__main__":
    main()
