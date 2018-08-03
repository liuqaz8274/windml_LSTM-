import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# 加载matplotlib工具包，使用该工具包可以对预测的sin函数曲线进行绘图
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')
from matplotlib import pyplot as plt
learn = tf.contrib.learn
HIDDEN_SIZE = 30  # Lstm中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔
# 定义生成正弦数据的函数
def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;第i+TIMESTEPS项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE,state_is_tuple=True)
    return lstm_cell

# 定义lstm模型
def lstm_model(X, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    
    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    
    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=0.1)
    return predictions, loss, train_op

# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="Models/model_2"))
# 生成数据
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))
# 拟合数据
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# 计算预测值
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
plt.show()