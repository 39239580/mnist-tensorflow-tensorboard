# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:19:52 2018

@author: Weixia
"""

import tensorflow as tf 
import os 

def variable_summary(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) #标准差
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

            
class BASE_CONV_MODULE:
    def __init__(self,kernel_size_x,kernel_size_y,num_channels,conv_deep,conv_stride,conv_padding):
        self.kernel_size_X=kernel_size_x
        self.kernel_size_Y=kernel_size_y
        self.num_channels=num_channels
        self.conv_deep=conv_deep
        self.conv_padding=conv_padding
        self.s_size=conv_stride
        #batch_normalization,在tensorflow提供的API接口按照封装程度
        #由低到高排列有：tf.nn.batch_normalization
        #tf.layers.batch_normalization
        #tf.contrib.batcg_normalization
        #参考 https://www.cnblogs.com/cloud-ken/p/8566948.html
        # https://blog.csdn.net/Leo_Xu06/article/details/79054326

    def base_conv(self,x,Training):
        input_tensor=self.input_tensor(x)
        with tf.variable_scope('conv_layer') as scope:  
            conv_weights=tf.get_variable('conv_weights',
                                         shape=[self.kernel_size_X,self.kernel_size_Y,self.num_channels,self.conv_deep],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv_baise=tf.get_variable('conv_baise',shape=[self.conv_deep],
                                       initializer=tf.constant_initializer(0.0))
            conv_op=tf.nn.conv2d(input_tensor,conv_weights,
                                 strides=[1,self.s_size,self.s_size,1],
                                 padding=self.conv_padding)
        pre_activate=tf.nn.bias_add(conv_op,conv_baise)
        pre_activate=tf.layers.batch_norm(pre_activate,decay=0.999,epsilon=0.001,is_training=Training)#使用BN层
        conv_out=tf.nn.relu(pre_activate)
        tf.summary.histogram('Preactivate',self.pre_activate)
        tf.summary.histogram('Activate',conv_out)
        variable_summary(self.conv_weights)
        variable_summary(self.conv_baise)
        return conv_out
    
class BASE_AVG_POOL:
    def __init__(self,pool_size,s_size,pool_padding):
        self.pool_size=pool_size
        self.s_size=s_size
        self.pool_padding=pool_padding
    def base_avg_pool(self,x):        
        pool_out=tf.nn.avg_pool(x,ksize=[1,self.pool_size,self.pool_size,1],
                                strides=[1,self.s_size,self.s_size,1],padding=self.pool_padding)       
        return pool_out
    
class BASE_MAX_POOL:
    def __init__(self,pool_size,s_size,pool_padding):
        self.pool_size=pool_size
        self.s_size=s_size
        self.pool_padding=pool_padding
    def base_avg_pool(self,x):        
        pool_out=tf.nn.max_pool(x,ksize=[1,self.pool_size,self.pool_size,1],
                                strides=[1,self.s_size,self.s_size,1],padding=self.pool_padding)
        tf.summary.histogram('Max_pool',pool_out)
        return pool_out
class BASE_FC:
    def __init__(self,INPUT_NODE,OUTPUT_NODE):
        self.input_node=INPUT_NODE
        self.output_node=OUTPUT_NODE
    def base_fc(input_tensor1,regularizer,train):
        fc_weight=tf.get_variable('fc_weight',shape=[self.input_node,self.output_node],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc_weight))
        fc_baises=tf.get_variable('fc_baises',shape=[OUTPUT_NODE],initializer=tf.constant_initializer(0.1))
        Preactivate=tf.matmul(input_tensor1,fc_weights)+fc_baises
        fc=tf.nn.relu(Preactivate)
        if train:
            fc =tf.nn.dropout(fc,keep_prob=Dropout_rate)
        variable_summary(fc_weight)
        variable_summary(fc_baises)
        tf.summary.histogram('Preactivate',Preactivate)
        tf.summary.histogram('Activate',fc)
        return fc

class INCPTION_Aux:  #辅助网络
    def __init__(self,in_channels,out_channels,out,).
    
class INCPETION_BASE_MODULE_v1:
    def __init__(self,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,stride_1x1,conving_padding_1x1,
                 kernel_size_3x3_1,num_channels_3x3_1,conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1,
                 kernel_size_3x3_2,conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2,
                 kernel_size_3x3_3,conv_deep_3x3_3,stride_3x3_3,conving_padding_3x3_3,
                 kernel_size_5x5_1,num_channels_5x5_1,conv_deep_5x5_1,stride_5x5_1,conving_padding_5x5_1,
                 kernel_size_5x5_2,conv_deep_5x5_2,stride_5x5_2,conving_padding_5x5_2,
                 pool_size,        s_size,            pool_padding,
                 kernel_size_2,    num_channels_2,    conv_deep_2,    stride_2,     conving_padding_2):

        self.branch1x1=BASE_CONV_MODULE(kernel_size_1x1,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,stride_1x1,conving_padding_1x1)

        self.branch3x3_1=BASE_CONV_MODULE(kernel_size_3x3_1,kernel_size_3x3_1,num_channels_3x3_1,
                                              conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1)
        self.branch3x3_2=BASE_CONV_MODULE(kernel_size_3x3_2,kernel_size_3x3_2,conv_deep_3x3_1,
                                              conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2)
        self.branch3x3_3=BASE_CONV_MODULE(kernel_size_3x3_3,kernel_size_3x3_3,conv_deep_3x3_2,
                                              conv_deep_3x3_3,stride_3x3_3,conving_padding_3x3_3)

        self.branch5x5_1=BASE_CONV_MODULE(kernel_size_5x5_1,kernel_size_5x5_1,num_channels_5x5_1,
                                              conv_deep_5x5_1,stride_5x5_1,conving_padding_5x5_1)
        self.branch5x5_2=BASE_CONV_MODULE(kernel_size_5x5_2,kernel_size_5x5_2,conv_deep_5x5_1,
                                              conv_deep_5x5_2,stride_5x5_2,conving_padding_5x5_2)

        self.branch_pool_1=BASE_AVG_POOL(pool_size,s_size,pool_padding)  #平均池化层
        self.branch_pool_2=BASE_CONV_MODULE(kernel_size_2,kernel_size_2,num_channels_2,
                                                conv_deep_2,stride_2,conving_padding_2)
        
    def forward(self,x,concat_dims,training):
        with tf.variable_scope('branch0'):
            branch1x1=self.branch1x1(x,Training=training)
        with tf.variable_scope('branch2'):
            branch3x3_1=self.branch3x3_1(x,Training=training)
            branch3x3_2=self.branch3x3_2(branch3x3_1,Training=training)
            branch3x3_3=self.branch3x3_3(branch3x3_2,Training=training)
        with tf.variable_scope('branch1'):
            branch5x5_1=self.branch5x5_1(x,Training=training)
            branch5x5_2=self.branch5x5_2(branch5x5_1,Training=training)
        with tf.variable_scope('branch2'):
            branch_pool=self.branch_pool_1(x)
            branch_pool=self.branch_pool_2(branch_pool,Training=training)
        
        outputs=tf.concat(concat_dim=concat_dims,[branch1x1,branch5x5_2,branch3x3_3,branch_pool])
        
        return outputs

class INCPETION_BASE_MODULE_v2:
    def __init__(self,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,stride_1x1,conving_padding_1x1,
                 kernel_size_7x7A_1,num_channels_7x7A_1,conv_deep_7x7A_1,stride_7x7A_1,conving_padding_7x7A_1,
                 kernel_size_7x7A_2_x,kernel_size_7x7A_2_y,conv_deep_7x7A_2,stride_7x7A_2,conving_padding_7x7A_2,
                 kernel_size_7x7A_3_x,kernel_size_7x7A_3_y,conv_deep_7x7A_3,stride_7x7A_3,conving_padding_7x7A_3,
                 
                 kernel_size_7x7B_1,num_channels_7x7B_1,stride_7x7B_1,conving_padding_7x7B_1,
                 kernel_size_7x7B_2_x,kernel_size_7x7B_2_y,conv_deep_7x7B_2,stride_7x7B_2,conving_padding_7x7B_2,
                 kernel_size_7x7B_3_x,kernel_size_7x7B_3_y,conv_deep_7x7B_3,stride_7x7B_3,conving_padding_7x7B_3,
                 kernel_size_7x7B_4_x,kernel_size_7x7B_4_y,conv_deep_7x7B_4,stride_7x7B_4,conving_padding_7x7B_4,
                 kernel_size_7x7B_5_x,kernel_size_7x7B_4_y,conv_deep_7x7B_5,stride_7x7B_5,conving_padding_7x7B_5,
                 
                 pool_size,        s_size,            pool_padding,
                 kernel_size_2,    num_channels_2,    conv_deep_2,    strid_2,   conving_padding_2):
        
        self.branch1x1=BASE_CONV_MODULE(kernel_size_1x1,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,conving_padding_1x1)
        
        
        self.branch7x7A_1=BASE_CONV_MODULE(kernel_size_7x7A_1,kernel_size_7x7A_1,num_channels_7x7A_1,
                                          conv_deep_7x7A_1,stride_7x7A_1,conving_padding_7x7A_1)
        self.branch7x7A_2=BASE_CONV_MODULE(kernel_size_7x7A_2_x,kernel_size_7x7A_2_y,conv_deep_7x7A_1,
                                          conv_deep_7x7A_2,stride_7x7A_2,conving_padding_7x7A_2)
        self.branch7x7A_3=BASE_CONV_MODULE(kernel_size_7x7A_3_x,kernel_size_7x7A_3_y,conv_deep_7x7A_2,
                                          conv_deep_7x7A_3,stride_7x7A_3,conving_padding_7x7A_3)
        
        self.branch7x7B_1=BASE_CONV_MODULE(kernel_size_7x7B_1,kernel_size_7x7B_1,num_channels_7x7B_1,
                                          conv_deep_7x7B_1,stride_7x7B_1,conving_padding_7x7B_1)
        self.branch7x7B_2=BASE_CONV_MODULE(kernel_size_7x7B_2_x,kernel_size_7x7B_2_y,conv_deep_7x7B_1,
                                          conv_deep_7x7B_2,stride_7x7B_2,conving_padding_7x7B_2)
        self.branch7x7B_3=BASE_CONV_MODULE(kernel_size_7x7B_3_x,kernel_size_7x7B_3_y,conv_deep_7x7B_2,
                                          conv_deep_7x7B_3,stride_7x7B_3,conving_padding_7x7B_3)
        self.branch7x7B_4=BASE_CONV_MODULE(kernel_size_5x5_4_x,kernel_size_7x7B_4_y,conv_deep_7x7B_3,
                                          conv_deep_7x7B_4,stride_7x7B_4,conving_padding_7x7B_4)
        self.branch7x7B_5=BASE_CONV_MODULE(kernel_size_7x7B_5_x,kernel_size_7x7B_5_y,conv_deep_7x7B_4,
                                          conv_deep_7x7B_5,stride_7x7B_5,conving_padding_7x7B_5)
        
        
        self.branch_pool_1=BASE_AVG_POOL(pool_size,s_size,pool_padding)  #平均池化层
        self.branch_pool_2=BASE_CONV_MODULE(kernel_size_2,kernel_size_2,num_channels_2,
                                            conv_deep_2,stride_2,conving_padding_2)
        
    def forward(self,x,concat_dims,training):
        with tf.variable_scope('branch0'):
            branch1x1=self.branch1x1(x,Training=training)
        with tf.variable_scope('branch1'):
            branch7x7A_1=self.branch7x7A_1(x,Training=training)
            branch7x7A_2=self.branch7x7A_2(branch7x7A_1,Training=training)
            branch7x7A_3=self.branch7x7A_3(branch7x7B_2,Training=training)
        with tf.variable_scope('branch2'):
            branch7x7B_1=self.branch7x7B_1(x,Training=training)
            branch7x7B_2=self.branch7x7B_2(branch7x7B_1,Training=training)
            branch7x7B_3=self.branch7x7B_3(branch7x7B_2,Training=training)
            branch7x7B_4=self.branch7x7B_4(branch7x7B_3,Training=training)
            branch7x7B_5=self.branch7x7B_5(branch7x7B_4,Training=training)
        with tf.variable_scope('branch3'):
            branch_pool=self.branch_pool_1(x)
            branch_pool=self.branch_pool_2(branch_pool,Training=training)
        
        outputs=tf.concat(concat_dim=concat_dims,[branch1x1,branch7x7A_3,branch7x7B_5,branch_pool])
        
        return outputs

class INCPETION_BASE_MODULE_v3:
    def __init__(self,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,stride_1x1,conving_padding_1x1,
                 kernel_size_3x3_1,num_channels_3x3_1,conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1,
                 kernel_size_3x3_2,conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2,
                 kernel_size_3x3_3,conv_deep_3x3_3,stride_3x3_3,conving_padding_3x3_3,
                 pool_size,    s_size,    pool_padding):
        self.branch1x1=BASE_CONV_MODULE(kernel_size_1x1,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,stride_1x1,conving_padding_1x1)
        
        self.branch3x3_1=BASE_CONV_MODULE(kernel_size_3x3_1,kernel_size_3x3_1,num_channels_3x3_1,
                                          conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1)
        self.branch3x3_2=BASE_CONV_MODULE(kernel_size_3x3_2,kernel_size_3x3_2,conv_deep_3x3_1,
                                          conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2)
        self.branch3x3_3=BASE_CONV_MODULE(kernel_size_3x3_3,kernel_size_3x3_3,conv_deep_3x3_2,
                                          conv_deep_3x3_3,stride_3x3_3,conving_padding_3x3_3)
        
        self.branch_pool_1=BASE_AVG_POOL(pool_size,s_size,pool_padding)  #平均池化层
        
    def forward(self,x,concat_dims,training):
        branch1x1=self.branch1x1(x,Training=training)
        
        branch3x3_1=self.branch3x3_1(x,Training=training)
        branch3x3_2=self.branch3x3_2(branch3x3_1,Training=training)
        branch3x3_3=self.branch3x3_3(branch3x3_2,Training=training)
        
        branch_pool=self.branch_pool_1(x)
        
        outputs=tf.concat(concat_dim=concat_dims,[branch1x1,branch3x3_3,branch_pool])
        
        return outputs
    
class INCPETION_BASE_MODULE_v4:
    def __init__(self,kernel_size_3x3_1,num_channels_3x3_1,conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1,
                 kernel_size_3x3_2,num_channels_3x3_2,conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2,
                 
                 kernel_size_7x7A_1,num_channels_7x7A_1,conv_deep_7x7A_1,stride_7x7A_1,conving_padding_7x7A_1,
                 kernel_size_7x7A_2_x,kernel_size_7x7A_2_y,conv_deep_7x7A_2,stride_7x7A_2,conving_padding_7x7A_2,
                 kernel_size_7x7A_3_x,kernel_size_7x7A_3_y,conv_deep_7x7A_3,stride_7x7A_3,conving_padding_7x7A_3,
                 kernel_size_7x7A_4_x,kernel_size_7x7A_4_y,conv_deep_7x7A_4,stride_7x7A_4,conving_padding_7x7A_4,
                 
                 pool_size,    s_size,    pool_padding):
        
        self.branch3x3_1=BASE_CONV_MODULE(kernel_size_3x3_1,kernel_size_3x3_1,num_channels_3x3_1,
                                          conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1)
        self.branch3x3_2=BASE_CONV_MODULE(kernel_size_3x3_2,kernel_size_3x3_2,conv_deep_3x3_1,
                                          conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2)
        
        
        self.branch7x7A_1=BASE_CONV_MODULE(kernel_size_7x7A_1,kernel_size_7x7A_1,num_channels_7x7A_1,
                                          conv_deep_7x7A_1,stride_7x7A_1,conving_padding_7x7A_1)
        self.branch7x7A_2=BASE_CONV_MODULE(kernel_size_7x7A_2_x,kernel_size_7x7A_2_y,conv_deep_7x7A_1,
                                          conv_deep_7x7A_2,stride_7x7A_2,conving_padding_7x7A_2)
        self.branch7x7A_3=BASE_CONV_MODULE(kernel_size_7x7A_3_x,kernel_size_7x7A_3_y,conv_deep_7x7A_2,
                                          conv_deep_7x7A_3,stride_7x7A_3,conving_padding_7x7A_3)
        self.branch7x7A_4=BASE_CONV_MODULE(kernel_size_7x7A_4_x,kernel_size_7x7A_4_y,conv_deep_7x7A_3,
                                          conv_deep_7x7A_4,stride_7x7A_4,conving_padding_7x7A_4)        
        
        self.branch_pool_1=BASE_AVG_POOL(pool_size,s_size,pool_padding)  #平均池化层
        
    def forward(self,x,concat_dims,training):
        with tf.variable_scope('branch0'):
            branch3x3_1=self.branch3x3_1(x,Training=training)
            branch3x3_2=self.branch3x3_2(branch7x7A_1,Training=training)
            
        with tf.variable_scope('branch1'):
            branch7x7A_1=self.branch7x7A_1(x,Training=training)
            branch7x7A_2=self.branch7x7A_2(branch7x7A_1,Training=training)
            branch7x7A_3=self.branch7x7A_3(branch7x7B_2,Training=training)
            branch7x7A_4=self.branch7x7A_4(branch7x7B_2,Training=training)        
        
        with tf.variable_scope('branch2'):
            branch_pool=self.branch_pool_1(x)
        
        outputs=tf.concat(concat_dim=concat_dims,[branch3x3_2,branch7x7A_3,branch_pool])
        
        return outputs

class INCPETION_BASE_MODULE_v5:
    def __init__(self,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,stride_1x1,conving_padding_1x1,
                 
                 kernel_size_7x7A_1,num_channels_7x7A_1,conv_deep_7x7A_1,stride_7x7A_1,conving_padding_7x7A_1,
                 kernel_size_7x7A_2_x,kernel_size_7x7A_2_y,conv_deep_7x7A_2,stride_7x7A_2,conving_padding_7x7A_2,
                 kernel_size_7x7A_3_x,kernel_size_7x7A_3_y,conv_deep_7x7A_3,stride_7x7A_3,conving_padding_7x7A_3,
                 
                 kernel_size_7x7B_1,num_channels_7x7B_1,conv_deep_7x7B_1,stride_7x7B_1,conving_padding_7x7B_1,
                 kernel_size_7x7B_2_x,kernel_size_7x7A_2_y,conv_deep_7x7B_2,stride_7x7B_2,conving_padding_7x7B_2,
                 kernel_size_7x7B_3_x,kernel_size_7x7A_3_y,conv_deep_7x7B_3,stride_7x7A_3,conving_padding_7x7B_3,
                 kernel_size_7x7B_4_x,kernel_size_7x7A_4_y,conv_deep_7x7B_4,stride_7x7A_4,conving_padding_7x7B_4,

                 pool_size,        s_size,            pool_padding,
                 kernel_size_2,    num_channels_2,    conv_deep_2,     stride_2,    conving_padding_2):
        
        self.branch1x1=BASE_CONV_MODULE(kernel_size_1x1,kernel_size_1x1,num_channels_1x1,conv_deep_1x1,conving_padding_1x1)
        
        
        self.branch3x3_1=BASE_CONV_MODULE(kernel_size_3x3_1,kernel_size_3x3_1,num_channels_3x3_1,
                                          conv_deep_3x3_1,stride_3x3_1,conving_padding_3x3_1)
        self.branch3x3_2=BASE_CONV_MODULE(kernel_size_3x3_2_x,kernel_size_3x3_2_y,conv_deep_3x3_1,
                                          conv_deep_3x3_2,stride_3x3_2,conving_padding_3x3_2)
        self.branch3x3_3=BASE_CONV_MODULE(kernel_size_3x3_3_x,kernel_size_3x3_3_y,conv_deep_3x3_2,
                                          conv_deep_3x3_3,stride_3x3_3,conving_padding_3x3_3)
        
        self.branch7x7B_1=BASE_CONV_MODULE(kernel_size_7x7B_1,kernel_size_7x7B_1,num_channels_7x7B_1,
                                          conv_deep_7x7B_1,stride_7x7B_1,conving_padding_7x7B_1)
        self.branch7x7B_2=BASE_CONV_MODULE(kernel_size_7x7B_2,kernel_size_7x7B_2,conv_deep_7x7B_1,
                                          conv_deep_7x7B_2,stride_7x7B_2,conving_padding_7x7B_2)
        self.branch7x7B_3=BASE_CONV_MODULE(kernel_size_7x7B_3_x,kernel_size_7x7B_3_y,conv_deep_7x7B_2,
                                          conv_deep_7x7B_3,stride_7x7B_3,conving_padding_7x7B_3)
        self.branch7x7B_4=BASE_CONV_MODULE(kernel_size_5x5_4_x,kernel_size_7x7B_4_y,conv_deep_7x7B_3,
                                          conv_deep_7x7B_4,stride_7x7B_4,conving_padding_7x7B_4)        
        
        self.branch_pool_1=BASE_AVG_POOL(pool_size,s_size,pool_padding)  #平均池化层
        self.branch_pool_2=BASE_CONV_MODULE(kernel_szie_2,kernel_szie_2,num_channels_2,
                                            conv_deep_2,stride_2,conving_padding_2)
        
    def forward(self,x,concat_dims,training):
        with tf.variable_scope('branch0'):
            branch1x1=self.branch1x1(x,Training=training)
        
        with tf.variable_scope('branch1'):
            branch3x3_1=self.branch3x3_1(x,Training=training)
            branch3x3_2=self.branch3x3_2(branch3x3_1,Training=training)
            branch3x3_3=self.branch3x3_3(branch3x3_1,Training=training)
            branch3x3_out=tf.concat(concat_dim=concat_dims,[branch3x3_2,branch3x3_3])
        
        with tf.variable_scope('branch2'):
            branch7x7_1=self.branch7x7_1(x,Training=training)
            branch7x7_2=self.branch7x7_2(branch7x7_1,Training=training)
            branch7x7_3=self.branch7x7_3(branch7x7_2,Training=training)
            branch7x7_4=self.branch7x7_4(branch7x7_2,Training=training)        
            branch7x7_out=tf.concat(concat_dim=concat_dims,[branch7x7_3,branch7x7_4])
        
        with tf.variable_scope('branch3'):
            branch_pool=self.branch_pool_1(x)
            branch_pool=self.branch_pool_2(branch_pool,Training=training)
        
        outputs=tf.concat(concat_dim=concat_dims,[branch1x1,branch3x3_out,branch7x7_out,branch_pool])
        
        return outputs
    
class INCEPTION_V3:
    def __init__(self):
        with tf.variable_scope('layer-1-conv'):
            self.Conv2d1=BASE_CONV_MODULE(3,3,32,2,'VALID')
        with tf.variable_scope('layer-2-conv'):
            self.Conv2d2=BASE_CONV_MODULE(3,32,32,1,'VALID')
        with tf.variable_scope('layer-3-conv'):
            self.Conv2d3=BASE_CONV_MODULE(3,32,64,1,'SAME')
        with tf.variable_scope('layer-4-pool'):
            self.Max_pool4=BASE_MAX_POOL(3,2,'VALID')
        with tf.variable_scope('layer-5-conv'):
            self.Conv2d5=BASE_CONV_MODULE(3,64,80,1,'VALID')
        with tf.variable_scope('layer-6-conv'):
            self.Conv2d6=BASE_CONV_MODULE(3,80,192,2,'VALID')
        with tf.variable_scope('layer-7-pool'):
            self.Conv2d7=BASE_CONV_MODULE(3,192,288,1,'SAME')
        with tf.variable_scope('layer-8-incption_module'):
            self.MIX_Block1_1=INCPETION_BASE_MODULE_v1(1,288,64,1,'SAME',1,288,64,1,'SAME',
                                                       3,96,1,'SAME',3,96,1,'SAME',
                                                       1,288,48,1,'SAME',5,64,1,'SAME',
                                                       3,1,'SAME',1,288,32,1,'SAME')   #输出为64+64+96+32=256
        with tf.variable_scope('layer-9-incption_module'):
            self.MIX_Block1_2=INCPETION_BASE_MODULE_v1(1,256,64,1,'SAME',1,256,64,1,'SAME',
                                                       3,96,1,'SAME',3,96,1,'SAME',
                                                       1,256,48,1,'SAME',5,64,1,'SAME',
                                                       3,1,'SAME',1,256,64,1,'SAME')   #输出为64+64+96+64=288
        with tf.variable_scope('layer-10-incption_module'):
            self.MIX_Block1_3=INCPETION_BASE_MODULE_v1(1,288,64,1,'SAME',1,288,64,1,'SAME',
                                                       3,96,1,'SAME',3,96,1,'SAME',
                                                       1,256,48,1,'SAME',5,64,1,'SAME',
                                                       3,1,'SAME',1,256,64,1,'SAME')   #输出为64+64+96+64=288)
        with tf.variable_scope('layer-11-incption_module'):
            self.MIX_Block2_1=INCPETION_BASE_MODULE_v3(3,288,384,2,'VALID',1,288,64,1,'',
                                                       3,96,1,'SAME',3,96,2,'SAME',
                                                       3,2,'VALID')#输出为384+96+288=768
        with tf.variable_scope('layer-12-incption_module'):
            self.MIX_Block2_2=INCPETION_BASE_MODULE_v2(1,768,192,1,'SAME',1,768,128,1,'SAME',1,7,128,128,1,'SAME',
                                                       7,1,128,192,1,'SAME',1,768,128,1,'SAME',7,1,128,128,1,'SAME',
                                                       1,7,128,128,1,'SAME',7,1,128,128,1,'SAME',1,7,192,192,'SAME',
                                                       3,1,'SAME',1,768,192,1,'SAME')     #输出为192+192+192+192=768
        with tf.variable_scope('layer-13-incption_module'):
            self.MIX_Block2_3=INCPETION_BASE_MODULE_v2(1,768,192,1,'SAME',1,768,160,1,'SAME',1,7,160,160,1,'SAME',
                                                       7,1,160,192,1,'SAME',1,768,160,1,'SAME',7,1,160,160,1,'SAME',
                                                       1,7,160,160,1,'SAME',7,1,160,160,1,'SAME',1,7,160,192,1,'SAME',
                                                       3,1,'SAME',1,768,192,1,'SAME')     #输出为192+192+192+192=768
        with tf.variable_scope('layer-14-incption_module'):
            self.MIX_Block2_4=INCPETION_BASE_MODULE_v2(1,768,192,1,'SAME',1,768,160,1,'SAME',1,7,160,160,1,'SAME',
                                                       7,1,160,192,1,'SAME',1,768,160,1,'SAME',7,1,160,160,1,'SAME',
                                                       1,7,160,160,1,'SAME',7,1,160,160,1,'SAME',1,7,160,192,1,'SAME',
                                                       3,1,'SAME',1,768,192,1,'SAME')     #输出为192+192+192+192=768
        with tf.variable_scope('layer-15-incption_module'):
            self.MIX_Block2_5=INCPETION_BASE_MODULE_v2(1,768,192,1,'SAME',1,768,192,1,'SAME',1,7,192,192,1,'SAME',
                                                       7,1,192,192,1,'SAME',1,768,192,1,'SAME',7,1,192,192,1,'SAME',
                                                       1,7,192,192,1,'SAME',7,1,192,192,1,'SAME',1,7,192,192,1,'SAME',
                                                       3,1,'SAME',1,768,192,1,'SAME')     #输出为192+192+192+192=768
        with tf.variable_scope('layer-16-incption_module'):
            self.MIX_Block3_1=INCPETION_BASE_MODULE_v4(1,768,192,1,'SAME',3,192,320,2,'VALID',1,768,192,1,'SAME',
                                                       1,7,192,192,1,'SAME',7,1,192,192,1,'SAME',3,192,192,2,'VALID',
                                                       3,2,'VALID') #输出为192+320+768=1280个通道
        with tf.variable_scope('layer-17-incption_module'):
            self.MIX_Block3_2=INCPETION_BASE_MODULE_v5(1,1280,320,1,'SAME',1,1280,384,1,'SAME',
                                                       1,3,384,384,1,'SAME',3,1,384,384,1,'SAME',
                                                       1,1280,448,1,'SAME',3,448,384,1,'SAME',
                                                       1,3,384,384,1,'SAME',3,1,384,384,1,'SAME',
                                                       3,1,'SAME',1,1280,192,1,'SAME') #输出为320+768+768+192=2048
        with tf.variable_scope('layer-18-incption_module'):
            self.MIX_Block3_3=INCPETION_BASE_MODULE_v5(1,1280,320,1,'SAME',1,1280,384,1,'SAME',
                                                       1,3,384,384,1,'SAME',3,1,384,384,1,'SAME',
                                                       1,1280,448,1,'SAME',3,448,384,1,'SAME',
                                                       1,3,384,384,1,'SAME',3,1,384,384,1,'SAME',
                                                       3,1,'SAME',1,1280,192,1,'SAME') #输出为320+768+768+192=2048
        with tf.variable_scope('layer-19'):
            self.Max_pool19=BASE_MAX_POOL(8,1,'VALID') #池化层
        with tf.variable_scope('layer-20'):
            self.FC20=BASE_FC(2048,1000) #全连接层及池化操作                        
            
    def inference(self,x,regularizer,train):
        input_tensor=x
        layer1=self.Conv2d1(x)
        layer2=self.Conv2d2(layer1)
        layer3=self.Conv2d3(layer2)
        layer4=self.Max_pool4(layer3)
        layer5=self.Conv2d5(layer4)
        layer6=self.Conv2d6(layer5)
        layer7=self.Max_pool7(layer6)
        layer8=self.MIX_Block1_1(layer8)
        layer9=self.MIX_Block1_2(layer9)
        layer10=self.MIX_Block1_3(layer10)
        layer11=self.MIX_Block2_1(layer11)
        layer12=self.MIX_Block2_2(layer12)
        layer13=self.MIX_Block2_3(layer13)
        layer14=self.MIX_Block2_4(laerr14)
        layer15=self.MIX_Block2_5(layer15)
        layer16=self.MIX_Block3_1(layer16)
        layer17=self.MIX_Block3_2(layer17)
        layer18=self.MIX_Block3_3(layer18)
        layer19=self.Max_pool19(layer19)
        layer20=self.FC20(layer19,reguluarizer,train)
        logits=layer20
        return logits
    
        


        
        

    
    
    
    
        
    
        
    
    