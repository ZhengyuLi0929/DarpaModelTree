"""

 run_tests.py  (author: Anson Wong / git: ankonzoid)

 Runs 3 tests to make sure our model tree works as expected.

"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_csv_data
from src.ModelTree import ModelTree
from sklearn.metrics import mean_squared_error as mse

def main():
    # ====================
    # Run sanity checks on model tree before training (using our own data)
    #  1) Reproduce model result on depth-0 model tree
    #  2) Reproduce sklearn DecisionTreeRegressor result using mean regression + mse
    #  3) Reproduce sklearn DecisionTreeClassifier result using modal class + gini loss
    # ====================
    #run_tests(ModelTree, os.path.join("data", "test_train.csv"))

    # ====================
    # For 1D polynomial data using a model tree with linear regression model
    # ====================

    # Generate 1D polynomial data and save as a csv
    '''
    func = lambda x: (x-1)*(x-4)*(x-8)*(x-8)
    data_csv_data_filename = os.path.join("data", "data_poly4_regr.csv")
    generate_csv_data(func, data_csv_data_filename,  x_range=(0, 10), N=500)

    # Read generated data
    X, y, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=True)
    assert X.shape[1] == 1
    '''
    nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected",
    "international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate",
    "maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage",
    "other/chavez","other/chavez/anti","protests","violence"]
    targets = ["twitter_event","twitter_user","twitter_newuser","youtube_event","youtube_user","youtube_newuser"]
    
    for target in targets:
        mapes = np.zeros(4)
        rmses = np.zeros(4)
        all_mape = 0
        all_rmse = 0
        for key in nodelist:
            name = key.replace('/','#')
            fname = name + "_bert_5day_8500_train.csv"
            data_csv_data_filename = os.path.join(target,fname)
            X, y, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            # Train different depth model tree fits and plot results
            #from models.mean_regr import mean_regr
            #plot_model_tree_fit(mean_regr(), X, y, name, mapes, rmses, target)
            from models.linear_regr import linear_regr
            plot_model_tree_fit(linear_regr(), X, y, name, mapes, rmses, target)
        mapes = mapes/18
        rmses = rmses/18
        for i in range(4):
            print("event: {}, depth: {}".format(target, i))
            print("ape:", round(mapes[i], 4))
            print("rmse:", round(rmses[i], 4))
            print("==================================")

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

def plot_model_tree_fit(model, X, y, name, mapes, rmses, target):
        output_filename = os.path.join("output_"+target, "bert_5day_8500_linear_{}_greedy_leaf_5_{}_fit.png".format(model.__class__.__name__, name))
        #print("Saving model tree predictions plot y vs x to '{}'...".format(output_filename))

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
            
            data_csv_data_filename = os.path.join(target, name+"_bert_5day_8500_test.csv")
            X_test, y_test, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            y_train_pred = model_tree.predict (X)
            y_pred = model_tree.predict(X_test)
            y_pred = postprocess(y_pred)
            rmse, ape = evaluation(y_test, y_pred)
            
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

        plt.suptitle('Model tree (model = {}) fits for different depths'.format(model.__class__.__name__), fontsize=25)
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

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

def run_tests(ModelTree, data_csv_filename):

    print("Running model tree tests...")
    eps = 1E-6 # tolerance for test acceptance
    X, y, header = load_csv_data(data_csv_filename, mode="regr")

    # Test 1
    print(" [1/3] Checking depth-0 model tree...")
    from models.linear_regr import linear_regr
    model = linear_regr()
    MTR_0 = ModelTree(model, max_depth=0, min_samples_leaf=20,
                      search_type="greedy", n_search_grid=100)
    loss_model = experiment(model, X, y)
    loss_MTR_0 = experiment(MTR_0, X, y)
    print("  -> loss(linregr)={:.6f}, loss(MTR_0_linregr)={:.6f}...".format(loss_model, loss_MTR_0))
    if np.abs(loss_model - loss_MTR_0) > eps:
        exit("err: passed test 1!")
    else:
        print("  -> passed test 1!")

    # Test 2
    print(" [2/3] Reproducing DecisionTreeRegressor sklearn (depth=20) result...")
    from models.mean_regr import mean_regr
    MTR = ModelTree(mean_regr(), max_depth=20, min_samples_leaf=10,
                    search_type="greedy", n_search_grid=100)
    from models.DT_sklearn_regr import DT_sklearn_regr
    DTR_sklearn = DT_sklearn_regr(max_depth=20, min_samples_leaf=10)
    loss_MTR = experiment(MTR, X, y)
    loss_DTR_sklearn = experiment(DTR_sklearn, X, y)
    print("  -> loss(MTR)={:.6f}, loss(DTR_sklearn)={:.6f}...".format(loss_MTR, loss_DTR_sklearn))
    if np.abs(loss_MTR - loss_DTR_sklearn) > eps:
        exit("err: passed test 2!")
    else:
        print("  -> passed test 2!")

    # Test 3
    print(" [3/3] Reproducing DecisionTreeClassifier sklearn (depth=20) result...")
    from models.modal_clf import modal_clf
    MTC = ModelTree(modal_clf(), max_depth=20, min_samples_leaf=10,
                    search_type="greedy", n_search_grid=100)
    from models.DT_sklearn_clf import DT_sklearn_clf
    DTC_sklearn = DT_sklearn_clf(max_depth=20, min_samples_leaf=10)
    loss_MTC = experiment(MTC, X, y)
    loss_DTC_sklearn = experiment(DTC_sklearn, X, y)
    print("  -> loss(MTC)={:.6f}, loss(DTC_sklearn)={:.6f}...".format(loss_MTC, loss_DTC_sklearn))
    if np.abs(loss_MTC - loss_DTC_sklearn) > eps:
        exit("err: passed test 3!")
    else:
        print("  -> passed test 3!")
    print()

def experiment(model, X, y):
    model.fit(X, y)  # train model
    y_pred = model.predict(X)
    loss = model.loss(X, y, y_pred)  # compute loss
    return loss

# Driver
if __name__ == "__main__":
    main()
