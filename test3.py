from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
import sys
import os
from windml.datasets.nrel import NREL
from windml.mapping.power_mapping import PowerMapping
from windml.model.windpark import Windpark
from windml.model.turbine import Turbine
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from tensorflow.python.framework import graph_util
#import os  
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
#start = time.clock()

cityname=['tehachapi','cheyenne','palmsprings','lasvegas','lancaster']
method=['KMeans','SpectralClustering','AgglomerativeClustering','Birch']
citynum=4
#int(sys.argv[1])
methodnum=0
#int(sys.argv[2]);
K=3
#int(sys.argv[3])
target_idx = NREL.park_id[cityname[citynum]]
year_from,year_to=2004,2006





park_id = NREL.park_id[cityname[citynum]]
windpark = NREL().get_windpark(park_id, 3, 2004,2006)

target = windpark.get_target()
feature_window, horizon = 3, 3
mapping = PowerMapping()   
data_1 = np.array(mapping.get_features_park(windpark, feature_window, horizon)) 
 
data_2 = np.array(mapping.get_labels_turbine(target, feature_window, horizon)).reshape(-1,1)
 
lendata=len(data_1)
data1 = data_1[:lendata:6]
data2 = data_2[:lendata:6]

half=train_end=int(math.floor(len(data1) * 0.5))




input_size=data1.shape[1] 
hidden_size=10     #隐藏层数        
output_size=1  
learning_rate_init=0.005
layer_num=5;   #rnn层数
tf.reset_default_graph() 






def get_data(batch_size=60,time_step=20,train_begin=0,train_end=half):  
    batch_index=[]  
          
    scaler_for_x=MinMaxScaler(feature_range=(0,1))    
    scaler_for_y=MinMaxScaler(feature_range=(0,1))  
    
    scaled_x_data=scaler_for_x.fit_transform(data1) 
    scaled_y_data=scaler_for_y.fit_transform(data2)  
    label_train = scaled_y_data[train_begin:train_end]  
    label_test = scaled_y_data[train_end:]  
    normalized_train_data = scaled_x_data[train_begin:train_end]  
    normalized_test_data = scaled_x_data[train_end:]  
      
    train_x,train_y=[],[]     
    for i in range(len(normalized_train_data)-time_step):  
        if i % batch_size==0:  
            batch_index.append(i)  
        x=normalized_train_data[i:i+time_step,:input_size]  
        y=label_train[i:i+time_step,np.newaxis]
        train_x.append(x.tolist())  
        train_y.append(y.tolist())  
    batch_index.append((len(normalized_train_data)-time_step))  
    size=(len(normalized_test_data)+time_step-1)//time_step     
    test_x,test_y=[],[]    
    for i in range(size-1):  
        x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]  
        y=label_test[i*time_step:(i+1)*time_step]  
        test_x.append(x.tolist())  
        test_y.extend(y)
        
    test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())  
    test_y.extend((label_test[(i+1)*time_step:]).tolist())      
      
    return batch_index,train_x,train_y,test_x,test_y,scaler_for_y  

def train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=half):  
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])  
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size,1])  
    batch_index,train_x,train_y,test_x,test_y,scaler_for_y = get_data(batch_size,time_step,train_begin,train_end)  
    test_y = scaler_for_y.inverse_transform(np.array(test_y).reshape(-1,1))  
    
    
    with tf.Graph().as_default():
            
            with tf.Session() as sess:  
                sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
     
                sess.run(tf.global_variables_initializer())  
                
                #########
                tf.saved_model.loader.load(sess, ['test'], 'model/model4')
                input_x = sess.graph.get_tensor_by_name("input:0")
                output = sess.graph.get_tensor_by_name("output:0")   
                ################
                
                start = time.clock();
                test_predict=list()  
                for step in range(len(test_x)-1):
                    prob=sess.run(output,feed_dict={input_x:[test_x[step]]})     
                    predict=prob.reshape((-1)) 
                    test_predict.extend(predict)  
                    
                test_predict = scaler_for_y.inverse_transform(np.array(test_predict).reshape(-1,1))  
                test_y=test_y[0:len(test_predict)]
        
                for i in range(len(test_predict)):
                    if test_predict[i] > 30.0:
                        test_predict[i] = 30.0
                    if test_predict[i] < 0.0:
                        test_predict[i] = 0.0
        
                elapsed = (time.clock() - start);
                
                mse =mean_squared_error(test_predict,test_y)
        
                print (elapsed,' feature_window:',feature_window,' mse:',mse,'scalermin ',scaler_for_y.min_,'scalerscale ',scaler_for_y.scale_) ; 
        
      
        
                plt.figure(figsize=(20,8))  
                plt.plot(data2[:len(data2)])  
                rk=[None for _ in range(half)] + [x for x in test_predict]
                plt.plot(rk[:len(rk)])  
                plt.xlim(len(test_y)+half-1500,len(test_y)+half-1000)
                plt.show() 
    return test_predict 

test_predict = train_lstm(batch_size=2000,time_step=15,train_begin=0,train_end=half) 

