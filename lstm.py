from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import time
from windml.datasets.nrel import NREL
from windml.mapping.power_mapping import PowerMapping
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import mean_squared_error

start = time.clock()


park_id = NREL.park_id['tehachapi']
windpark = NREL().get_windpark(park_id, 3, 2004,2006)
target = windpark.get_target()

feature_window, horizon = 3, 3
mapping = PowerMapping()
data1 = np.array(mapping.get_features_park(windpark, feature_window, horizon))
data2 = np.array(mapping.get_labels_turbine(target, feature_window, horizon)).reshape(-1,1)


#目标多向前取feature_window个点
#data1 = np.array(mapping.get_features_park(windpark, feature_window*2, horizon))
#data2 = np.array(mapping.get_labels_turbine(target, feature_window*2, horizon)).reshape(-1,1)
#region = int(data1.shape[1]/(feature_window*2))
#for i in range(region-1):
#    for j in range(feature_window):
#        data1=np.delete(data1,[(i+1)*feature_window],axis = 1)
#
#

lendata=len(data1)

data1 = data1[:lendata:5]
data2 = data2[:lendata:5]

half=train_end=int(math.floor(len(data1) * 0.5))



hidden_size=10     #隐藏层数
input_size=data1.shape[1]        
output_size=1  
learning_rate_init=0.01
layer_num=2;   #rnn层数
tf.reset_default_graph() 
 
weights={  
         'in':tf.Variable(tf.random_normal([input_size,hidden_size])),  
         'out':tf.Variable(tf.random_normal([hidden_size,1]))  
         }  
biases={  
        'in':tf.Variable(tf.constant(0.1,shape=[hidden_size,])),  
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))  
        }  
keep_prob = tf.placeholder(tf.float32)





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

def lstm(X):    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']  
    b_in=biases['in']    
    input=tf.reshape(X,[-1,input_size])   
    input_rnn=tf.matmul(input,w_in)+b_in  
    input_rnn=tf.reshape(input_rnn,[-1,time_step,hidden_size])
    
    lstm_cell_1=tf.contrib.rnn.LSTMCell(num_units=hidden_size ,use_peepholes=True,activation=tf.nn.softsign ,forget_bias=1.0, state_is_tuple=True)
#    lstm_cell_2=tf.contrib.rnn.LSTMCell(num_units=hidden_size ,use_peepholes=True,activation=tf.nn.softsign ,forget_bias=1.0, state_is_tuple=True)   

#    lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_1, input_keep_prob=1.0, output_keep_prob=0.99)
    
#    m_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    
    init_state_1 = lstm_cell_1.zero_state(batch_size,dtype=tf.float32) 
#    init_state_2 = lstm_cell_2.zero_state(batch_size,dtype=tf.float32)  
#    (output_rnn, final_states) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_1,lstm_cell_2,input_rnn, initial_state_fw=init_state_1,initial_state_bw=init_state_2,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(lstm_cell_1, input_rnn,initial_state=init_state_1, dtype=tf.float32)   
#    
#    output=tf.reshape(output_rnn,[-1,hidden_size])  
    final_states=tf.reshape(output_rnn,[-1,hidden_size]) 
    w_out=weights['out']  
    b_out=biases['out']
    
#    pred=tf.matmul(output,w_out)+b_out 
    pred2=tf.matmul(final_states,w_out)+b_out
    return pred2,final_states  

def train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=half):  
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])  
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size,1])  
    batch_index,train_x,train_y,test_x,test_y,scaler_for_y = get_data(batch_size,time_step,train_begin,train_end)  
    pred,_=lstm(X) 
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))  #MSE
    
    global_step = tf.Variable(0) 
    lr=tf.train.exponential_decay(learning_rate_init,global_step,decay_steps=len(train_x)/batch_size,decay_rate=0.995,staircase=False)      
    
    train_op=tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step) 
    
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())  
        
        iter_time = 100
        
            
        for i in range(iter_time):
#            
            for step in range(len(batch_index)-1): 
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})  
            if i % 10 == 0:    
                lr_=sess.run(lr)
                print('iter:',i,'loss:',loss_ , ' lr ' ,lr_) 
                
                
        test_predict=[]  
        for step in range(len(test_x)-1):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})     
            predict=prob.reshape((-1))  
            test_predict.extend(predict)  
        test_predict = scaler_for_y.inverse_transform(np.array(test_predict).reshape(-1,1))  
        test_y = scaler_for_y.inverse_transform(np.array(test_y).reshape(-1,1))  
        test_y=test_y[0:len(test_predict)]
        
#        for i in range(len(test_predict)):
#            if test_predict[i] > 30.0:
#                test_predict[i] = 30.0
#            if test_predict[i] < 0.0:
#                test_predict[i] = 0.0
        
        
        #输出结果
        mse =mean_squared_error(test_predict,test_y)
        print ( 'mse:',mse)

        fo = open("lstm.out", "a")
        str1=str(iter_time)+" use peepholes  "+str(layer_num)+' loss:'+str(loss_)+ 'mse:'+str(mse)
        fo.write (str1)
        # 关闭打开的文件
        fo.close()
    return test_predict 

test_predict = train_lstm(batch_size=1000,time_step=20,train_begin=0,train_end=half) 

elapsed = (time.clock() - start)
print("Time used:",elapsed) 