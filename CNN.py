# -*- coding: utf-8 -*-
"""
@Data：
@Copyright: Weixia
@author: Weixia
@softwaretool：Pycharm
@version:2017
"""

import tensorflow as tf
#import numpy as np
from numpy.random import RandomState
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHES = 5000
w1 = tf.Variable(tf.random_normal([2,3],seed=1,stddev=1)) # 创建变量 ，定义神经网络参数
w2 = tf.Variable(tf.random_normal([3,1],seed=1,stddev=1))
x = tf.placeholder(tf.float32,shape=[None,2],name='x_input')   # 占位符
y_ = tf.placeholder(tf.float32,shape=[None,1],name='y_input')

a = tf.matmul(x,w1)     # 定义前向传播过程
y = tf.matmul(a,w2)

y = tf.sigmoid(y)   # 定义损失函数与反向传播的算法

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y , 1e-10 , 1.0))+
                                (1-y)*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

rdm=RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2 < 1)for (x1, x2) in X]]
with tf.Session() as sess:
    init_opt=tf.global_variables_initializer()
    sess.run(init_opt)
    print(sess.run(w1))
    print(sess.run(w2))
    for epoch in range(EPOCHES):
        start=(epoch*BATCH_SIZE)%dataset_size
        end=min(start+BATCH_SIZE,dataset_size)
        sess.run(train_step, feed_dict = {x: X[start : end], y_: Y[start : end]})
        if epoch %1000==0:
            total_cross_entropy=sess.run(cross_entropy , feed_dict={x : X , y_ : Y})
            print("After %d training step(s),cross entropy on all data is %g"%(epoch,total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))






