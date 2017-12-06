# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:54:51 2017

@author: gewushengping
"""

import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
df = pd.read_csv("C:/SPB_Data/creditcard.csv")
df.head()
df.describe()
v_features = df.ix[:,1:29].columns
'''
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()
'''
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)
#二值化之后的数据放到后面了，相当于新的变量，没有删掉之前的数据
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0
#Rename 'Class' to 'Fraud'.
df = df.rename(columns={'Class': 'Fraud'})
Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]

X_train = Fraud.sample(frac=0.8)
count_Frauds = len(X_train)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

# X_test contains all the transaction not in X_train.
X_test = df.loc[~df.index.isin(X_train.index)]#随机抽了之后把剩下的当做测试集
#这样划分测试集
#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)#搅乱顺序
X_test = shuffle(X_test)
#Add our target features to y_train and y_test.
y_train = X_train.Fraud
y_train = pd.concat([y_train, X_train.Normal], axis=1)#拼接成了两列了
y_test = X_test.Fraud
y_test = pd.concat([y_test, X_test.Normal], axis=1)
#Drop target features from X_train and X_test.
#训练数据删掉标签
X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)

#Check to ensure all of the training/testing dataframes are of the correct length
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

ratio = len(X_train)/count_Frauds*1.25 #这里增加一些权重试试
y_train.Fraud *= ratio
y_test.Fraud *= ratio
features = X_train.columns.values
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std
split = int(len(y_test)/2)#测试数据有28000多其中一半作为验证集了
inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
inputX_valid = X_test.as_matrix()[:split]#测试数据的一半用于验证，一半用于测试
inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()[split:]
inputY_test = y_test.as_matrix()[split:]
input_nodes = 36#这里的意思是37个变量
# Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 1.5 
# Number of nodes in each hidden layer
hidden_nodes1 = 18
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)
# Percent of nodes to keep during dropout.
pkeep = tf.placeholder(tf.float32)#这里使用了一个占位符
x = tf.placeholder(tf.float32, [None, input_nodes])#占位符定义了列的维度
#数据
# layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = 0.15))
#w1的参数数量
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)#矩阵乘法x在前面
#sigmod 本身就和矩阵没有关系
#输入多少就输出多少
# layer 2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = 0.15))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
# layer 3
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev = 0.15)) 
b3 = tf.Variable(tf.zeros([hidden_nodes3]))
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
y3 = tf.nn.dropout(y3, pkeep)#对激活值随机drop_out部分值
# layer 4
W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev = 0.15)) 
b4 = tf.Variable(tf.zeros([2]))

y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)#最后一层使用softmax
# output
y = y4
print ('y4\n',y4)
y_ = tf.placeholder(tf.float32, [None, 2])#
training_epochs = 1000 # should be 2000, it will timeout when uploading
training_dropout = 0.6
display_step = 1 # 10 
n_samples = y_train.shape[0]
batch_size = 30 #一共60000
learning_rate = 0.01
# Cost function: Cross Entropy
cost = -tf.reduce_sum(y_ * tf.log(y))#损失函数
#加权是体现在这里了
# We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Correct prediction if the most likely value (Fraud or Normal) from softmax equals the target value.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Note: some code will be commented out below that relate to saving/checkpointing your model.
accuracy_summary = [] # Record accuracy values for plot
cost_summary = [] # Record cost values for plot
valid_accuracy_summary = [] 
valid_cost_summary = [] 
stop_early = 0 # To keep track of the number of epochs before early stopping
# Save the best weights so that they can be used to make the final predictions
checkpoint = "./best_model.ckpt"
saver = tf.train.Saver(max_to_keep=1)
# Initialize variables and tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in range(training_epochs): #
        for batch in range(int(n_samples/batch_size)):#这里是批处理的个数
            batch_x = inputX[batch*batch_size : (1+batch)*batch_size]
            batch_y = inputY[batch*batch_size : (1+batch)*batch_size]
            #参数之间是连带的
         #  dfli=sess.run (y4,feed_dict={x: batch_x, 
         #                                    y_: batch_y,#这里输入的是两列
         #                                    pkeep: training_dropout})
         #   print('dfli\n',dfli)
            sess.run([optimizer], feed_dict={x: batch_x, 
                                             y_: batch_y,#这里输入的是两列
                                             pkeep: training_dropout})            
        
        # Display logs after every 10 epochs
        if (epoch) % display_step == 0:#每一次都输出
            train_accuracy, newCost, = sess.run([accuracy, cost], feed_dict={x: inputX, 
                                                                            y_: inputY,
                                                                            pkeep: training_dropout})

            valid_accuracy, valid_newCost,validcorreec_pr = sess.run([accuracy, cost,correct_prediction], feed_dict={x: inputX_valid, 
                                                                                  y_: inputY_valid,
                                                                                  pkeep: 1})

            print ("Epoch:", epoch,
                   "Acc =", "{:.5f}".format(train_accuracy), 
                   "Cost =", "{:.5f}".format(newCost),
                   "Valid_Acc =", "{:.5f}".format(valid_accuracy), 
                   "Valid_Cost = ", "{:.5f}".format(valid_newCost))
            # Save the weights if these conditions are met.
            if epoch > 0 and valid_accuracy > max(valid_accuracy_summary) and valid_accuracy > 0.993:
                saver.save(sess, checkpoint)
            # Record the results of the model
            accuracy_summary.append(train_accuracy)
            cost_summary.append(newCost)
            valid_accuracy_summary.append(valid_accuracy)
            valid_cost_summary.append(valid_newCost)
            # If the model does not improve after 15 logs, stop the training.
            #如果有15次小于最大的就停止训练

            if valid_accuracy < max(valid_accuracy_summary) and epoch > 80:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0
            
    print()
    print("Optimization Finished!")
    print()   
    # Find the predicted values, then use them to build a confusion matrix
predicted = tf.argmax(y, 1)
with tf.Session() as sess:  
#    # Load the best weights
    saver.restore(sess, checkpoint)    
    testing_predictions, testing_accuracy = sess.run([predicted, accuracy], 
                                                     feed_dict={x: inputX_test, y_:inputY_test, pkeep: 1})    
    print("F1-Score =", f1_score(inputY_test[:,1], testing_predictions))
    print("Testing Accuracy =", testing_accuracy)
    print()
    c = confusion_matrix(inputY_test[:,1], testing_predictions)
    print("confusion_matrix\n",c)    
    training_predictions, training_accuracy = sess.run([predicted, accuracy], 
                                                     feed_dict={x: inputX, y_:inputY, pkeep: 1})    
    c1 = confusion_matrix(inputY[:,1], training_predictions)
    print("confusion_matrix\n",c1)
    #show_confusion_matrix(c, ['Fraud', 'Normal'])
   # confusion_matrix
   #[[   39     2]
   #[  134 28306]]
  




