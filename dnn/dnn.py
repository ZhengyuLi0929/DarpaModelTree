from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import json
import os

import tensorflow.compat.v1 as tf
import tensorflow.math as tm
import tensorflow.keras.backend as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from training_index import training_index
from sklearn.metrics import mean_squared_error as mse
tf.disable_v2_behavior()
# from tensorflow import keras
# from tensorflow.keras import layers


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
TK_SILENCE_DEPRECATION=1

FLAGS = None

def evaluation(Y_test, Y_pred):
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test) + 0.1), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / sum(Y_test)
    return rmse, ape
    
def postprocess(pred):
    pred = np.array([int(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred

class dnn:
    def __init__(self, xdims, ydims, log_file, x_for_loss, y_for_loss, platform, k, n):
        self.platform = platform
        self._xdims = xdims
        self._ydims = ydims
        self.x_placeholder = tf.placeholder(tf.float32, [None, xdims])
        self.y_placeholder = tf.placeholder(tf.float32, [None, ydims])
        self.ind_placeholder = tf.placeholder(tf.int32, [None, ])

        self.scale = tf.ones(shape=[18,1])
        self.scale = tf.Variable(self.scale, name = "scale")
        
        self.keep_prob = tf.placeholder(tf.float32)
        self.y_pred = self.deepnn(self.x_placeholder, self.ind_placeholder)

        self.loss = tf.sqrt(tf.losses.mean_squared_error(self.y_placeholder, self.y_pred))
        #self.loss = tf.sqrt(tf.losses.mean_squared_error(tm.cumsum(self.y_placeholder)/(tk.sum(self.y_placeholder) + 0.1), tm.cumsum(self.y_pred)/(tk.sum(self.y_pred) + 0.1)))

        self.log_file = log_file
        self.x_for_loss = x_for_loss
        self.y_for_loss = tf.constant(y_for_loss, dtype=tf.float32)
        self.normalized_rmse = tf.sqrt(tf.losses.mean_squared_error(tm.cumsum(self.y_for_loss)/(tk.sum(self.y_for_loss) + 0.1), tm.cumsum(self.y_pred)/(tk.sum(self.y_pred) + 0.1)))
        self.train_loss = tf.sqrt(tf.losses.mean_squared_error(self.y_for_loss, self.y_pred))
        #self.loss_mape = np.mean(np.abs(self.y_for_loss.eval() - self.y_pred.eval()) / self.y_for_loss.eval())
        
        self.k = k
        self.n = n

    #def calc_mape(self):
        
    def deepnn(self, x, ind):
        """deepnn builds the graph for a deep net for classifying digits.
        Args:
        x: an input tensor with the dimensions (N_examples, xdims).
        Returns:
        A tuple y. y is a tensor of shape (N_examples, ydims)
        """
        '''
        print(x.shape)
        indices = ind
        self.ind = ind
        depth = 18
        oh = tf.one_hot(indices, depth)
        self.oh = oh

        s = tf.matmul(oh, self.scale)
        self.s = s
        
        '''
        
        
        #h_layer1 = tf.layers.dense(inputs=x, units=1, name='h0', activation=tf.nn.relu)
        #h_layer0 = tf.layers.dense(inputs=x, units=100, name='h0', activation=tf.nn.relu)
        #h_layer0_drop = tf.nn.dropout(h_layer0, self.keep_prob, name='h0_drop')
        #h_layer1 = tf.layers.dense(inputs = h_layer0, units = 1, name = 'h1', activation = tf.nn.relu)
        
        stop1 = int(self._xdims/3)
        x1 = x[:,:stop1]
        x2 = x[:,stop1:2*stop1]
        x3 = x[:,2*stop1:3*stop1]
        #x4 = x[:,3*stop1:4*stop1]
        #x5 = x[:,4*stop1:]
        d1 = tf.layers.dense(inputs=x1, units=100, name='h0_1', activation=tf.nn.relu)
        d1_layer0_drop = tf.nn.dropout(d1, self.keep_prob, name='d1_drop')
        d2 = tf.layers.dense(inputs=x2, units=100, name='h0_2', activation=tf.nn.relu)
        d2_layer0_drop = tf.nn.dropout(d2, self.keep_prob, name='d2_drop')
        d3 = tf.layers.dense(inputs=x3, units=100, name='h0_3', activation=tf.nn.relu)
        d3_layer0_drop = tf.nn.dropout(d3, self.keep_prob, name='d3_drop')
        #d4 = tf.layers.dense(inputs=x2, units=100, name='h0_4', activation=tf.nn.relu)
        #d4_layer0_drop = tf.nn.dropout(d4, self.keep_prob, name='d2_drop')
        #d5 = tf.layers.dense(inputs=x3, units=100, name='h0_5', activation=tf.nn.relu)
        #d5_layer0_drop = tf.nn.dropout(d5, self.keep_prob, name='d3_drop')
        
        layer_2 = tf.concat([d1,d2,d3],1)#,d4,d5], 1)
        h_layer1 = tf.layers.dense(inputs=layer_2, units=1, name='output', activation=tf.nn.relu)
        
        '''
        out = tf.multiply(s,h_layer1)
        self.out = out
        self.h0 = h_layer1
        return out
        '''
        return h_layer1
        
        
    def train(self, train_x, train_y, train_ind, IDs, asp, plt, batch_size=500, learn_rate=0.001):
        
        update_var_list = []
        tvars = tf.trainable_variables()
        for tvar in tvars:
            if "scale" not in tvar.name:
                update_var_list.append(tvar)
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, var_list=update_var_list)
        train_all = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

        train_index = training_index(train_x.shape[0])

        saver = tf.train.Saver()
        loss_test_rmse = []
        loss_train_rmse = []
        loss_train_mape = []
        loss_test_mape = []
        val_i = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            i = 0
            num_cal = 0
            init_loss = 0
            while True:
                index = train_index.next(batch_size)

                x = train_x[index]
                
                # normal train
                sess.run(train_step, feed_dict={
                    self.x_placeholder: x,
                    self.y_placeholder: train_y[index],
                    self.keep_prob: 0.5}
                    )
                
                '''
                # train scale vector
                if (i % 200 > 180 and plt == "twitter") or (i % 20 > 16 and plt == "youtube"):
                    train_vec = sess.run(train_all, feed_dict={
                    self.x_placeholder: x,
                    self.y_placeholder: train_y[index],
                    self.ind_placeholder: train_ind[index],
                    self.keep_prob: 0.5}
                    )
                else:
                    _, oh, s = sess.run([train_step,self.oh, self.s], feed_dict={
                    self.x_placeholder: x,
                    self.y_placeholder: train_y[index],
                    self.ind_placeholder: train_ind[index],
                    self.keep_prob: 0.5}
                    )
                '''
                if (i%200==0) or (i%40 == 0 and plt == "youtube"):
                    num_cal += 1
                    # training loss
                    
                    # no scale
                    train_loss, N_rmse = sess.run([self.train_loss, self.normalized_rmse], feed_dict={
                        self.x_placeholder: self.x_for_loss,
                        self.keep_prob: 1.0}
                        )
                    
                    '''
                    # auto trained
                    #print(oh.shape)
                    #print(s.flatten())
                    
                    train_loss, scale, N_rmse, out = sess.run([self.train_loss, self.scale, self.normalized_rmse, self.out], feed_dict={
                        self.x_placeholder: self.x_for_loss,
                        self.ind_placeholder: train_ind,
                        self.keep_prob: 1.0}
                    )
                    '''
                    if i == 0:
                        init_loss = train_loss
                    print(' step %d, train loss %g' % (i, train_loss))
                    print(' step %d, train rmse %g' % (i, N_rmse))
                    
                    # keep track of the loss
                    loss_train_rmse.append(train_loss)
                    
                    #print(' step %d, train mape loss %g' % (i, train_mape))
                    #loss_train_mape.append(train_mape)
                    
                    #print(scale.flatten())
                    
                    # prediction loss
                    saver.save(sess, self.log_file)
                    
                    test_loss_mse = 0
                    test_loss_mape = 0
                    test_loss_mape_nonzero = 0
                    for infoID in IDs:
                        '''
                        #BAD = ["arrests/opposition", "maduro/narco", "arrests/opposition/media", "crisis/looting",\
                               "maduro/illegitimate", "other/censorship_outage", "international/emigration",\
                               "violence/against_opposition/protesters", "international/break_us_relations",\
                               "maduro/events/pro", "maduro/legitimate/international", "maduro/events/anti",\
                               "other/request_observers", "assembly/illegitimate"]
                        #REALLY_BAD = ["maduro/legitimate/international", "maduro/events/anti", "other/request_observers",\
                                      "assembly/illegitimate"]
                        #if infoID in BAD:
                        #    continue
                        '''
                        infoID = infoID.replace('/','#')
                        test_x = np.loadtxt('./test/'+self.platform+'/'+infoID+'_bert_'+asp+'_131_x.csv', delimiter=',')
                        test_y = np.loadtxt('./test/'+self.platform+'/'+infoID+'_bert_'+asp+'_131_y.csv', delimiter=',')
                        #====
                        test_ind = np.loadtxt('./test/'+platform+'/'+infoID+'_bert_'+asp+'_131_ind.csv', delimiter=',')
                        #====
                        
                        pred_y = self.predict(test_x, test_ind).flatten()
                        pred_y = postprocess(pred_y)
                        
                        # evaluation
                        rmse, ape = evaluation(test_y, pred_y)
                        test_loss_mse += rmse
                        test_loss_mape += ape
                        
                        '''
                        if infoID == "violence" or infoID == "protests":
                            print(pred_y)
                        
                        this_id_loss = np.sum(np.square(test_y - pred_y))
                        test_loss_mse += this_id_loss
                        
                        
                        
                        #cumulative
                        mape_pred = []
                        mape_test = []
                        p = 0
                        t = 0
                        for j in range(14):
                            p += pred_y[j]
                            t += test_y[j]
                            if p == 0:
                                mape_pred.append(1)
                            else:
                                mape_pred.append(p)
                            if t == 0:
                                mape_test.appe(1)
                                
                            mape_test.append(t)
                        
                        test_nonzero = []
                        pred_nonzero = []
                        for j in range(14):
                            if test_y[j] == 0:
                                test_y[j] = 1
                            else:
                                test_nonzero.append(test_y[j])
                                pred_nonzero.append(pred_y[j])
                        test_nonzero = np.array(test_nonzero)
                        pred_nonzero = np.array(pred_nonzero)
                        this_mape_loss_nonzero = np.mean(np.abs(pred_nonzero - test_nonzero) / test_nonzero)
                        test_loss_mape_nonzero += this_mape_loss_nonzero
                        
                        this_mape_loss = np.mean(np.abs(pred_y - test_y) / test_y)
                        test_loss_mape += this_mape_loss
                        '''
                    
                    '''
                    test_loss = np.sqrt(test_loss_mse/(33 * 14))
                    print(' step %d, test loss %g' % (i, test_loss))
                    loss_test_rmse.append(test_loss)
                    
                    test_loss_mape = test_loss_mape / 33
                    loss_test_mape.append(test_loss_mape)
                    print(' step %d, test loss mape %g' % (i, test_loss_mape))
                    test_loss_mape_nonzero = test_loss_mape_nonzero / 33
                    print(' step %d, test loss mape (nonzero): %g' % (i, test_loss_mape_nonzero))
                    # save model for later usage
                    
                    path = "./checkpoints/multi/"+self.platform +"/"+str(self.k)+"_"+str(self.n)+"/"
                    if not os.path.exists(path):
                        os.mkdir(path)
                    saver.save(sess, path+str(i))
                    '''
                    
                    print(' step %d, test loss rmse %g' % (i, round (test_loss_mse/18, 4)))
                    print(' step %d, test loss mape %g' % (i, round (test_loss_mape/18, 4)))
                    
                    # save i value for plotting
                    val_i.append(i)
                    
                
                # manual stop
                #if (i == 1000) or (i == 160 and plt == "youtube"):
                # auto stop
                if num_cal > 2 and loss_train_rmse[-1]  < 0.5*init_loss:
                    break
                i += 1
            saver.save(sess, self.log_file)
                
        #with open('./loss/'+self.platform+'/loss_train.csv', 'w') as fout:
        #    for x in loss_train:
        #        fout.write(str(x)+'\n')
        '''
        plt.figure()
        plt.plot(val_i, loss_train)
        plt.savefig('./figures/train_loss_three_to_one'+self.platform+'.png')
        print(loss_train)
        plt.figure()
        plt.plot(val_i, loss_test)
        plt.savefig('./figures/test_loss_three_to_one'+self.platform+'.png')
        print(loss_test)
        '''
    def plot_loss(self):
        x = np.loadtxt('./loss/'+self.platform+'/loss_train.csv')
        plt.figure()
        plt.plot(x)
        plt.savefig('./figures/train_loss_'+self.platform+'.png')

    def predict(self, data_init, data_ind):
        saver = tf.train.Saver()
        #print (data_ind)
        ind = data_ind
        x = data_init

        with tf.Session() as sess:
            saver.restore(sess, self.log_file)
            Y = self.y_pred.eval(feed_dict={
                self.x_placeholder: x,
                #====
                self.ind_placeholder: ind,
                #===
                self.keep_prob: 1.0}
                )

        return Y


# main
options =["twitter_event","twitter_user","twitter_newuser","youtube_event","youtube_user","youtube_newuser"]
infoIDs_twitter = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected",
"international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate",
"maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage",
"other/chavez","other/chavez/anti","protests","violence"]

tt = {}
yt = {}
outt = {}
outy = {}
for ids in infoIDs_twitter:
    tt[ids] = dict()
    yt[ids] = dict()
    c = ['EventCount', 'UserCount','NewUserCount']
    for item in c:
        yt[ids][item] = {}
        tt[ids][item] = {}
for choice in options:

    #infoIDs_twitter = [u'guaido/illegitimate', u'protests', u'maduro/events/anti', u'assembly/legitimate', u'maduro/russia_support', u'international/military', u'arrests', u'international/respect_sovereignty', u'maduro/cuba_support', u'international/aid_rejected', u'violence', u'maduro/narco', u'other/planned_coup', u'crisis', u'other/restore_democracy', u'guaido/us_support', u'international/break_us_relations', u'arrests/opposition/media', u'other/anti_socialism', u'maduro/legitimate', u'international/emigration', u'crisis/looting', u'other/chavez/anti', u'maduro/dictator', u'arrests/opposition', u'other/request_observers', u'international/aid', u'maduro/illegitimate', u'maduro/events/pro', u'arrests/opposition/protesters', u'guaido/legitimate', u'crisis/lack_essentials', u'maduro/events', u'international/us_sanctions', u'maduro/illegitimate/international', u'guaido/legitimate/international', u'violence/against_opposition', u'violence/against_opposition/protesters', u'violence/against_maduro', u'maduro/legitimate/international', u'other/media_bias', u'other/chavez/pro', u'other/chavez', u'other/censorship_outage', u'military', u'military/desertions', u'assembly/illegitimate']
    k = -2
    n = 5

    platform = choice.split('_')[0]
    asp = choice.split('_')[1]

    train_x = np.loadtxt('./train/'+platform+'/'+str(n)+'/bert_'+asp+'_131_x.csv', delimiter=',')
    train_ind = np.loadtxt('./train/'+platform+'/'+str(n)+'/bert_'+asp+'_131_ind.csv')
    train_y = np.loadtxt('./train/'+platform+'/'+str(n)+'/bert_'+asp+'_131_y.csv', delimiter=',')
    train_y = train_y.reshape(len(train_y),1)
    print(train_x.shape)
    
    tf.reset_default_graph()
    pr = dnn(train_x.shape[1], train_y.shape[1], './log/'+platform+'/', train_x, train_y, platform, k, n)
    pr.train(train_x, train_y, train_ind, infoIDs_twitter, asp, platform)
    #pr.plot_loss()
    rmse = 0
    ape = 0
    for infoID in infoIDs_twitter:
        #BAD = ["arrests/opposition", "maduro/narco", "arrests/opposition/media", "crisis/looting", "maduro/illegitimate", "other/censorship_outage", "international/emigration", "violence/against_opposition/protesters", "international/break_us_relations", "maduro/events/pro", "maduro/legitimate/international", "maduro/events/anti", "other/request_observers", "assembly/illegitimate"]
        #REALLY_BAD = ["maduro/legitimate/international", "maduro/events/anti", "other/request_observers", "assembly/illegitimate"]
        #if infoID in BAD:
            #continue
        key = infoID
        infoID = infoID.replace('/','#')
        test_x = np.loadtxt('./test/'+platform+'/'+infoID+'_bert_'+asp+'_131_x.csv', delimiter=',')
        test_ind = np.loadtxt('./test/'+platform+'/'+infoID+'_bert_'+asp+'_131_ind.csv', delimiter=',')
        test_y = np.loadtxt('./test/'+platform+'/'+infoID+'_bert_'+asp+'_131_y.csv', delimiter=',')
        test_p = np.loadtxt('./test/'+platform+'/'+infoID+'_bert_'+asp+'_131_y_prev.csv', delimiter=',')
        pred_y = pr.predict(test_x, test_ind).flatten()
        pred_y = postprocess(pred_y)

        # print(test_y, type(test_y))
        # print(pred_y, type(pred_y))
        rmse_, ape_ = evaluation(test_y, pred_y)
        ape += ape_
        rmse += rmse_
        # base = datetime.datetime.today()
        '''
        base = datetime.datetime.strptime("2019/02/01", "%Y/%M/%d")
        t1 = [base + datetime.timedelta(days=x) for x in range(0,14)]
        t2 = [base + datetime.timedelta(days=x) for x in range(-39, 14)]

        newinfoID = infoID.replace('#','/')
        plt.figure(figsize=(8,3))
        # plt.xlabel('Time')
        plt.ylabel('Event #')
        plt.title('Twitter narritive -- \''+newinfoID+'\'')
        plt.gcf().autofmt_xdate()
        plt.plot(t2, np.concatenate([test_p,test_y]), linewidth = '3', label='GT')
        plt.plot(t1, pred_y, linewidth = '3', label='Pred')
        
        #this_err = abs(test_y - pred_y)
        #err += this_err
        #print (infoID, this_err, sum(this_err))
        
        plt.legend(loc='upper left')
        path = './figures/results/'+platform+'/' + str(k) +'_'+str(n)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path+'/'+asp+'_'+infoID+'_to_1_31_dnn.png')
        '''
        sdate = 1548979200000 # 2-1
        if asp == 'event':
            option = 'EventCount'
        elif asp == 'user':
            option = 'UserCount'
        else:
            option = 'NewUserCount'
        
        if platform == "twitter":
            for i in range(len(pred_y)):
                tt[key][option][str(sdate + i * 86400000)] = int(pred_y[i])
        else:
            for i in range(len(pred_y)):
                yt[key][option][str(sdate + i * 86400000)] = int(pred_y[i])
    
        outt[key] = pd.DataFrame(tt[key]).to_json()
        outy[key] = pd.DataFrame(yt[key]).to_json()
    #print ("total error:", err)
    #print ("err sum:", sum(err))

with open('youtube_UIUC_HYBRID_TEXT_.json', 'w') as outfile:
    json.dump(outy, outfile)
with open('twitter_UIUC_HYBRID_TEXT_.json', 'w') as outfile:
    json.dump(outt, outfile)
    
