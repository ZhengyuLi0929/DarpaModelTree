"""

 run_tests.py  (author: Anson Wong / git: ankonzoid)

 Runs 3 tests to make sure our model tree works as expected.

"""
import copy
import os, csv
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_csv_data
from src.ModelTree import ModelTree
from sklearn.metrics import mean_squared_error as mse

# ********************************
#
# seperate model (one model for one narrative)
#
# ********************************
def sepmod():
    
    nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected",
    "international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate",
    "maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage",
    "other/chavez","other/chavez/anti","protests","violence"]
    targets = ["twitter_event","twitter_user","twitter_newuser"]#,"youtube_event","youtube_user","youtube_newuser"]
    
    for target in targets:
        mapes = np.zeros(4)
        rmses = np.zeros(4)
        sizes = np.zeros(4)
        all_mape = 0
        all_rmse = 0
        decay = 0.85
        for key in nodelist:
            name = key.replace('/','#')
            fname = name + "_bert_1day_0_train.csv"
            data_csv_data_filename = os.path.join(target,fname)
            X, y, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            # decay
            Y = copy.deepcopy(X)
            for i in range(len(X)):
                if i == 0:
                    continue
                X[i] = X[i] - X[i-1] + decay *
            # Train different depth model tree fits and plot results
            #from models.mean_regr import mean_regr
            #plot_model_tree_fit(mean_regr(), X, y, name, mapes, rmses, target)
            from models.linear_regr import linear_regr
            plot_model_tree_fit(linear_regr(), X, y, name, mapes, rmses, sizes, target)
        mapes = mapes/18
        rmses = rmses/18
        for i in range(4):
            print("event: {}, depth: {}".format(target, i))
            #print("ape:", round(mapes[i], 4))
            #print("rmse:", round(rmses[i], 4))
            print("ape:", mapes[i])
            print("rmse:", rmses[i])
            print("==================================")

# ********************************
#
# one big linear model for all narratives
#
# ********************************
#def bigmod()
    



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

# ********************************
#
# Side functions for seperate model
#
# ********************************
def plot_model_tree_fit(model, X, y, name, mapes, rmses, sizes, target):
        '''
        # plot graph
        output_filename = os.path.join("output_"+target, "bert_1day_0_linear_{}_greedy_leaf_5_{}_fit.png".format(model.__class__.__name__, name))
        #print("Saving model tree predictions plot y vs x to '{}'...".format(output_filename))
        '''

        plt.figure(figsize=(20, 10))
        figure_str = "23"
        mape_ls = np.zeros(12)
        for depth in range(4):
            # Form model tree
            #print(" -> training model tree depth={}...".format(depth))
            model_tree = ModelTree(model, max_depth=depth, min_samples_leaf=5,
                                   search_type="greedy", n_search_grid=10)

            # Train model tree
            model_tree.fit(X, y, verbose=False)
            
            data_csv_data_filename = os.path.join(target, name+"_bert_1day_0_test.csv")
            X_test, y_test, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            y_train_pred = model_tree.predict (X)
            y_pred = model_tree.predict(X_test)
            y_pred = postprocess(y_pred)
            rmse, ape = evaluation(y_test, y_pred)
            
            sizes[depth] += sum(y_pred)
            #if name == "arrests":
            #    explanation = model_tree.explain(X, header)
            # calc mape
            '''
            m_pred = []
            m_truth = []
            for i in range(len(y_test)):
                if y_test[i] == 0:
                    continue
                m_truth.append(y_test[i])
                if y_pred[i] < 0:
                    y_pred[i] = 0
                    m_pred.append(0)
                    continue
                m_pred.append(y_pred[i])
            m_pred = np.array(m_pred)
            m_truth = np.array(m_truth)
            mape = round(np.mean(np.abs(m_pred - m_truth) / m_truth), 3)
            '''
            mapes[depth] += ape
            rmses[depth] += rmse
            '''
            # Plot predictions
            plt.subplot(int(figure_str + str(depth + 1)))
            plt.plot( np.concatenate((y, y_test),axis = None), markersize=5, color='k')
            plt.plot( np.concatenate((y_train_pred, y_pred), axis = None), markersize=5, color='r')
            
            #print(y_pred)
            #print(y_test)
            #plt.plot(y, markersize=5, color='k')
            #plt.plot(y_train_pred, markersize=5, color='r')
            # hidden: X[:, 0], y_pred...
            plt.legend(['data', 'fit'])
            plt.title("depth = {}, mape = {}, rmse = {}".format(depth, round(ape, 4), round(rmse, 4)))
            plt.xlabel("# days", fontsize=15)
            plt.ylabel("y", fontsize=15)
            plt.grid()
            '''
        '''
        plt.suptitle('Model tree (model = {}) fits for different depths'.format(model.__class__.__name__), fontsize=25)
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()
        '''
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
# ********************************
#
# Side functions for big model
#
# ********************************



# Driver
if __name__ == "__main__":
    sepmod()
    #bigmod()
