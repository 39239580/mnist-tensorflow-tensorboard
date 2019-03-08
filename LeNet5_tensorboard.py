# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:33:01 2018

@author: Weixia
"""

import tensorflow as tf
import os
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
#from multiprocessing import Process
#import multiprocessing

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
NUM_EPOCHES = 10000
MOVING_AVERAGE_DECAY = 0.99
#模型路径设置
MODEL_SAVE_PATH = './MNIST_data/tensorboard/model'
MODEL_NAME = 'model.ckpt'
# 数据设置
MAP_SIZE=28
CHANNELS=1


EVAL_INTERVAL_SECS=2

tf.reset_default_graph() #重置默认计算图，清除节点。

def BASE_CONV(kernel_size,num_channels,conv_deep,input_tensor,s_size,padding):
    conv_weights = tf.get_variable('conv_weights',shape=[kernel_size,kernel_size,num_channels,conv_deep],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv_baises  = tf.get_variable('conv_baises',shape=[conv_deep],initializer=tf.constant_initializer(0.0))
    conv_op=tf.nn.conv2d(input_tensor,conv_weights,strides=[1,s_size,s_size,1],padding=padding)
    conv_out=tf.nn.relu(tf.nn.bias_add(conv_op,conv_baises))
    variable_summaries(conv_weights)
    variable_summaries(conv_baises)
    tf.summary.histogram('Preactivate',tf.nn.bias_add(conv_op,conv_baises)) #直方图对激活函数前的分布进行统计 
    tf.summary.histogram('Activate',conv_out) #直方图对激活后的分布进行统计
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
    variable_summaries(fc_baises) #对 FC层的偏置行统计
    variable_summaries(fc_weight)  #对FC层的权重进行统计
    tf.summary.histogram('Activate',fc) #多FC层的激活后的进行直方图进行统计
    tf.summary.histogram('Preactivate',tf.matmul(input_tensor1,fc_weight)+fc_baises)
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
#inference中定义的常量与和前向传播函数不需要改变，前向传播通过tf.variable_scope实现了计算节点按照网络结构的划分
def train(input_data):
 
    with tf.name_scope('input_data'):
        x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, MAP_SIZE, MAP_SIZE, CHANNELS], name = 'x_input')
        y_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE, OUTPUT_NODE,], name='y_output')
        tf.summary.image('input_image',x,10)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZITION_RATE)
    y = inference(x,train,regularizer)
    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，在一般的情况下，训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0,trainable=False) #滑动平均，指数衰减型学习率中使用较多
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    with tf.name_scope('moving_averages'):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step) #初始变量
        # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量（比如global_step）就不需要了。
        # tf。trainable_variables返回的就是图上集合
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        loss =cross_entropy_mean +tf.add_n(tf.get_collection('losses'))#损失函数（损失值）
        tf.summary.scalar('cross_entory',cross_entropy_mean)
        tf.summary.scalar('loss',loss)
    with tf.name_scope('train_step'):
        learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                 global_step,
                                                 input_data.train.num_examples/BATCH_SIZE,
                                                 LEARNING_RATE_DECAY)#滑动平均的方式
        train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        tf.summary.scalar('learning_rate',learning_rate)
        # 即一次完成多个操作,每过一遍数据既要通过反向传播来更新神经网络中的参数。
        # 又要更新每一个参数的滑动平均值。是为了一次完成多个操作，等价于
        # train_op=tf.group([train_step,variable_average_op])
        # 为了一次完成多个操作,tf.control_dependencies与tf.group两种机制是一样的。
        with tf.control_dependencies([train_step,variables_averages_op]):
            train_op=tf.no_op('train')
    merged=tf.summary.merge_all() #将所有摘要合并，并写到日志里面,自动化配置功能。
    #初始化Tensoflow持久化类
    saver =tf.train.Saver()
    
    with tf.Session() as sess:
        #出事化写日志的writer，并将当前tensorflow写入日志  
        #在tf.summary.FileWriter(logdir,tf.get_default_graph())或写成
        #tf.summary.FileWriter（logdir, sess.graph）,第一种是可以卸载任何一种里面的
        #第二种是一般卸载sess中的。
        train_writer=tf.summary.FileWriter('./MNIST_data/tensorboard/log/train',sess.graph)
        tf.global_variables_initializer().run()
        for epoch in range (NUM_EPOCHES):
            xs,ys=input_data.train.next_batch(BATCH_SIZE)
            reshaped_xs=np.reshape(xs,(BATCH_SIZE,
                                       MAP_SIZE,
                                       MAP_SIZE,
                                       CHANNELS))
            #-----------*********添加计算成本的配置信息*****-------------#
            #配置运行时需要记录的信息
            run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #运行时记录运行信息的proto,metadata元数据。
            run_metadata=tf.RunMetadata()
            summary,_,loss_value,step=sess.run([merged,train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys},
                                       options=run_options,run_metadata=run_metadata)
            # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每个节点的时间，空间的开销信息
            #用于后期的计算性能改善
            train_writer.add_run_metadata(run_metadata,'step%d'%(epoch))
            train_writer.add_summary(summary,epoch)
            if epoch % 100 == 0: #epoch是从0开始，而step是从1开始                
                # %g。浮点数字（根据值的大小采用%e或%f），%e为采用可续计数法，用E代替e
                print("After %d training step(s)，loss on training batch is%g."%(step,loss_value))
                #os.path.join  将MODEL_NAME 加入到path路径下。
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
    #tensorboard日志功能

    train_writer.close()
                
                
def evaluate( input_data ):
    with tf.name_scope('evaluate_data'):   ##上下文文件管理机制with，
        # placeholder 占位符
        x = tf.placeholder(tf.float32,[input_data.validation.num_examples,MAP_SIZE,MAP_SIZE,CHANNELS],name='x-input')
        y_= tf.placeholder(tf.float32,[input_data.validation.num_examples,OUTPUT_NODE],name='y-output')
        tf.summary.image('evluate_data',x,10)
    with tf.name_scope('reshape_evaluate'):
        validate_feed={x:np.reshape(input_data.validation.images,(input_data.validation.num_examples,
                                                             MAP_SIZE,
                                                             MAP_SIZE,
                                                             CHANNELS)),
                                                             y_: input_data.validation.labels}
    with tf.name_scope('inference_evaluate'):
        y=inference(x_tensor=x,train=False,regularizer=None)
    with tf.name_scope('accuarcy_evaluate'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        # argmax（array，0）表示array按照列找出最大值那个的索引，
        # argmax(array,1) 表示array按照行找最大值那个的索引。
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #tf.cast整形数据转成浮点型数据成32位。
        tf.summary.scalar('accuracy',accuracy)
    with tf.name_scope('averages_evaluate'):
        variable_averages =tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
    merged=tf.summary.merge_all()
    saver=tf.train.Saver(variables_to_restore)
    while True:
        with tf.Session() as sess:
            evaluate_writer=tf.summary.FileWriter('./MNIST_data/tensorboard/log/evaluate',sess.graph)
            ckpt=tf.train.get_checkpoint_state(
                    MODEL_SAVE_PATH) #加载模型
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                #  通过文件名得到模型保存时迭代的轮数。
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]                
                summary,accuracy_score=sess.run([merged,accuracy], feed_dict=validate_feed,)
                print("After %s training step(s),validation accuracy = %g"%(global_step,accuracy_score))
            else:
                print("No checkpoint file found")
                return
            evaluate_writer.add_summary(summary,global_step)
            time.sleep(EVAL_INTERVAL_SECS)
    evaluate_writer.close()

def test (input_data):
    with tf.name_scope('test_data'):   ##上下文文件管理机制with，设置默认图为g
        # placeholder 占位符
        x = tf.placeholder(tf.float32,[input_data.test.num_examples,MAP_SIZE,MAP_SIZE,CHANNELS],name='x-input')
        y_= tf.placeholder(tf.float32,[input_data.test.num_examples,OUTPUT_NODE],name='y-output')
    with tf.name_scope('reshape_test'):
        test_feed={x:np.reshape(input_data.test.images,(input_data.test.num_examples,
                                                             MAP_SIZE,
                                                             MAP_SIZE,
                                                             CHANNELS)),
                                                             y_: input_data.test.labels}
    with tf.name_scope('inferenece_test'):
        y=inference(x_tensor=x,train=False,regularizer=None)
    with tf.name_scope('accuarcy_test'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        # argmax（array，0）表示array按照列找出最大值那个的索引，
        # argmax(array,1) 表示array按照行找最大值那个的索引。
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #tf.cast整形数据转成浮点型数据成32位。
    with tf.name_scope('moving_averages_evaluate'):
        variable_averages =tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
    saver=tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state(
                MODEL_SAVE_PATH) #加载模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            #通过文件名得到模型保存时迭代的轮数。
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]                
            accuracy_score=sess.run(accuracy, feed_dict=test_feed)
            print("After %s training step(s),test accuracy = %g"%(global_step,accuracy_score))
        else:
            print("No checkpoint file found")

#生成变量监控信息，并定义生成监控信息日志的操作。其中var给出了需要记录的张量，
#name给出了在可视化结果中显示的图标名称，这个名称一般与变量名一致。
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)  #标准差
        tf.summary.scalar('max',tf.reduce_max(var))  #最大值
        tf.summary.scalar('min',tf.reduce_min(var))  #最小值
        tf.summary.histogram('histogram',var)  #'直方图'
        #或者在定义的时候   def variable_summaries(var,name)时
        #tf.scalar_summary('mean/'+name,mean),作用与tf.summary_scalar('mean',mean)是一致的
        
        
        
        
        
                

def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data/DATA",one_hot=True)
    train(mnist)
#    print('The number of CPU is:'+str(multiprocessing.cpu_count())) #显示核心个数
#    e=multiprocessing.Event()
#    p1=Process(target=train,args=(mnist,))#旧版本中的用法，
    #p1=Process(target=train(mnist)) #新版本中的用法，实例化进程
#    p2=Process(target=evaluate,args=(mnist,))
    #p2=Process(target=evaluate(mnist))
    #p3=Process(target=test(mnist))
    #p1.start() #开启第一个进程
    #p2.start() #开启第二个进程
    #p3.start() #开启第三个进程
    
if __name__=='__main__':
    tf.app.run()
