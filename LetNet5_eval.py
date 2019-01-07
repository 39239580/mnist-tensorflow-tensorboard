# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:33:17 2018

@author: Weixia
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import LeNet5

EVAL_INTERVAL_SECS=2

def evaluate( mnist ):
    with tf.Graph().as_default() as g:   ##上下文文件管理机制with，设置默认图为g
        x = tf.placeholder(tf.float32,[mnist.validation.num_examples,LeNet5.MAP_SIZE,LeNet5.MAP_SIZE,LeNet5.CHANNELS],name='x-input')
        y_= tf.placeholder(tf.float32,[mnist.validation.num_examples,LeNet5.OUTPUT_NODE],name='y-output')
        validate_feed={x:np.reshape(mnist.validation.images,(mnist.validation.num_examples,
                                                             LeNet5.MAP_SIZE,
                                                             LeNet5.MAP_SIZE,
                                                             LeNet5.CHANNELS)),
                                                             y_: mnist.validation.labels}
        
        y=LeNet5.inference(x_tensor=x,train=False,regularizer=None)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        # argmax（array，0）表示array按照列找出最大值那个的索引，
        # argmax(array,1) 表示array按照行找最大值那个的索引。
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #tf.cast整形数据转成浮点型数据成32位。
        variable_averages =tf.train.ExponentialMovingAverage(LeNet5.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(
                        LeNet5.MODEL_SAVE_PATH) #加载模型
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
#                    通过文件名得到模型保存时迭代的轮数。
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]                
                    accuracy_score=sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s),validation accuracy = %g" %(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)
            
def main(argv = None):            
    mnist = input_data.read_data_sets("./MNIST_data/DATA",one_hot=True)
    evaluate(mnist)
    
if __name__ == '__main__':
    tf.app.run()
