# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:30:13 2018

@author: Weixia
"""

import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#网络结构参数
INPUT_NODE = 784
OUTPUT_NODE = 10

CONV1_SIZE =5 #卷积核的大小
NUM1_CHANNELS =1
CONV1_DEEP =32
POOL1_SIZE =2
POOL1_STRIDE =2

CONV2_SIZE =5
NUM2_CHANNELS =32
CONV2_DEEP =64
POOL2_SIZE =2
POOL2_STRIDE =2

FC5_NODE =512
FC6_NODE =10
DR5 =0.5

STRIDE1 =1
STRIDE2 =1

#超参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.999
REGULARAZITION_RATE = 0.0001
NUM_EPOCHES = 30000
MOVING_AVERAGE_DECAY = 0.99
#模型路径设置
MODEL_SAVE_PATH = './MNIST_data/model'
MODEL_NAME = 'model.ckpt'
# 数据设置
MAP_SIZE=28
CHANNELS=1

mnist = input_data.read_data_sets("./MNIST_data/DATA",one_hot=True)

tf.reset_default_graph() #重置默认计算图，清除节点。

def BASE_CONV(kernel_size,num_channels,conv_deep,input_tensor,s_size,padding):
    conv_weights = tf.get_variable('conv_weights',shape=[kernel_size,kernel_size,num_channels,conv_deep],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv_baises  = tf.get_variable('conv_baises',shape=[conv_deep],initializer=tf.constant_initializer(0.0))
    conv_op=tf.nn.conv2d(input_tensor,conv_weights,strides=[1,s_size,s_size,1],padding=padding)
    conv_out=tf.nn.relu(tf.nn.bias_add(conv_op,conv_baises))
    return conv_out

def BASE_POOL(kernel_size,s_size,input_tensor,padding):
    pool_out=tf.nn.max_pool( input_tensor,ksize=[1,kernel_size,kernel_size,1],strides=[1,s_size,s_size,1],padding=padding)
    return pool_out
    
def BASE_FC(input_tensor1,input_tensor2,FC_NODE,regularizer,train,DROPOUT_RATE,end_flags):
    fc_weight =tf.get_variable('fc_weight',shape=[input_tensor2,FC_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
#    if DROPOUT_RATE==0:
#        keep_prob=1.0
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(fc_weight))
    fc_baises =tf.get_variable('fc_baises',shape=[FC_NODE],initializer=tf.constant_initializer(0.1))
    if end_flags==True:  #是否是最后一层，若是，不需要做relu处理
        fc=tf.matmul(input_tensor1 ,fc_weight)+fc_baises
    else: #不是最后一层
        fc=tf.nn.relu(tf.matmul(input_tensor1, fc_weight)+fc_baises)
        if train:
            fc=tf.nn.dropout(fc,keep_prob=DROPOUT_RATE)
    return fc
        
def inference(x_tensor,train,regularizer):
    with tf.variable_scope('layer1_conv'):
        conv1=BASE_CONV(kernel_size=CONV1_SIZE, num_channels=NUM1_CHANNELS, conv_deep=CONV1_DEEP,
                        input_tensor=x_tensor, s_size=STRIDE1, padding='SAME')
        pool2_out=BASE_POOL(kernel_size=POOL1_SIZE, s_size=POOL1_STRIDE, input_tensor=conv1, padding='SAME')
        
    with tf.variable_scope('layer2_conv'):
        conv2=BASE_CONV(kernel_size=CONV2_SIZE, num_channels=NUM2_CHANNELS, conv_deep=CONV2_DEEP,
                        input_tensor=pool2_out, s_size=STRIDE2, padding='SAME')
        pool4_out=BASE_POOL(kernel_size=POOL2_SIZE, s_size=POOL2_STRIDE, input_tensor=conv2, padding='SAME')
    
    pool4_shape=pool4_out.get_shape().as_list()#拉直成一个向量  pool2_out 张量为4维的 [batch_size, 7,7,64]
    nodes = pool4_shape[1]*pool4_shape[2]*pool4_shape[3]
    reshaped = tf.reshape(pool4_out, [pool4_shape[0], nodes]) #batch_size=pool4_shape[0]
    
    with tf.variable_scope('layer3_fc'):
        fc1 = BASE_FC(input_tensor1=reshaped,input_tensor2=nodes,FC_NODE = FC5_NODE,regularizer=regularizer,train=train,DROPOUT_RATE=DR5,end_flags=False)
        
    with tf.variable_scope('layer4_fc'):
        fc2 = BASE_FC(input_tensor1=fc1,input_tensor2=FC5_NODE, FC_NODE = FC6_NODE,regularizer=regularizer,train=train,DROPOUT_RATE=1.0,end_flags=True)
        logit=fc2
    
    return logit

def train(input_data):
    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, MAP_SIZE, MAP_SIZE, CHANNELS], name = 'x_input')
    y_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE, OUTPUT_NODE,], name='y_output')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZITION_RATE)
    y = inference(x,train,regularizer)
    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，在一般的情况下，训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0,trainable=False) #滑动平均，指数衰减型学习率中使用较多
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step) #初始变量
    # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量（比如global_step）就不需要了。
    # tf。trainable_variables返回的就是图上集合
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss =cross_entropy_mean +tf.add_n(tf.get_collection('losses'))#损失函数（损失值）
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                             global_step,
                                             mnist.train.num_examples/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)#滑动平均的方式
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # 即一次完成多个操作,每过一遍数据既要通过反向传播来更新神经网络中的参数。
    # 又要更新每一个参数的滑动平均值。是为了一次完成多个操作，等价于
    # train_op=tf.group([train_step,variable_average_op])
    # 为了一次完成多个操作,tf.control_dependencies与tf.group两种机制是一样的。
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op('train')
    
    #初始化Tensoflow持久化类
    saver =tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range (NUM_EPOCHES):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs=np.reshape(xs,(BATCH_SIZE,
                                       MAP_SIZE,
                                       MAP_SIZE,
                                       CHANNELS))
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            if epoch % 100 == 0: #epoch是从0开始，而step是从1开始
                # %g。浮点数字（根据值的大小采用%e或%f），%e为采用可续计数法，用E代替e
                print("After %d training step(s)，loss on training batch is%g."%(step,loss_value))
                #os.path.join  将MODEL_NAME 加入到path路径下。
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
                

def main(argv=None):
    train(mnist)
    
if __name__=='__main__':
    tf.app.run()