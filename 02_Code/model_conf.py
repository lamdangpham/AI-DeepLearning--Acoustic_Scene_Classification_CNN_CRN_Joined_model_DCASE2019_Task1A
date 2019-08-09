import tensorflow as tf
import numpy as np
import os

from model_para import *

from cnn_bl_time_conf import *
from cnn_bl_freq_conf import *
from cnn_bl_conf import *

from dnn_bl01_conf import *
from dnn_bl02_conf import *

from nn_basic_layers import *
from rnn_para import *

#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()
        self.rnn_para   = rnn_para()

        # ============================== Fed Input
        self.input_layer_val     = tf.placeholder(tf.float32, [None, self.model_para.n_freq, self.model_para.n_time, self.model_para.n_chan], name="input_layer_val")
        self.input_layer_val_f01  = self.input_layer_val[:,0:128,  :,:]  #log-Mel spec
        self.input_layer_val_f02  = self.input_layer_val[:,128:256,:,:]  #gammatone spec
        self.input_layer_val_f03  = self.input_layer_val[:,256:384,:,:]  #CQT spec

        self.expected_classes    = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")
        self.mode                = tf.placeholder(tf.bool, name="running_mode")

        self.seq_len             = tf.placeholder(tf.int32, [None], name="seq_len" ) # for the dynamic RNN

        #============================== NETWORK CONFIGURATION
        # Call Batchnorm
        with tf.device('/gpu:0'), tf.variable_scope("fist_batch_norm_01")as scope:
             self.input_layer_val_01 = tf.contrib.layers.batch_norm(self.input_layer_val_f01, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        with tf.device('/gpu:0'), tf.variable_scope("fist_batch_norm_02")as scope:
             self.input_layer_val_02 = tf.contrib.layers.batch_norm(self.input_layer_val_f02, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        with tf.device('/gpu:0'), tf.variable_scope("fist_batch_norm_03")as scope:
             self.input_layer_val_03 = tf.contrib.layers.batch_norm(self.input_layer_val_f03, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        #====================================== RNN brand ============================================#     
        # Call CNN Time and Get CNN Time output
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01_01")as scope:
            self.cnn_time_ins_01 = cnn_bl_time_conf(self.input_layer_val_01, self.mode)
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01_02")as scope:
            self.cnn_time_ins_02 = cnn_bl_time_conf(self.input_layer_val_02, self.mode)
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01_03")as scope:
            self.cnn_time_ins_03 = cnn_bl_time_conf(self.input_layer_val_03, self.mode)

        self.cnn_time_ins_01_output = self.cnn_time_ins_01.final_output
        self.cnn_time_ins_02_output = self.cnn_time_ins_02.final_output
        self.cnn_time_ins_03_output = self.cnn_time_ins_03.final_output

        # Add all output time-CNN
        self.add_time = self.cnn_time_ins_01_output + self.cnn_time_ins_02_output + self.cnn_time_ins_03_output

        # Reshape CNN to feed into RNN
        [nS, nF, nT, nC] = self.add_time.get_shape()
        self.add_time_reshape = tf.reshape(self.add_time, [-1, nT*nF, nC])  #nS:nT:nC


        # Call b-RNN
        with tf.device('/gpu:0'), tf.variable_scope("bidirection_recurrent_layer_01") as scope:
            self.fw_cell_01, self.bw_cell_01 = bidirectional_recurrent_layer(self.rnn_para.n_hidden,
                                                             self.rnn_para.n_layer,
                                                             #seq_len=self.config.n_step,
                                                             #is_training=self.istraining,
                                                             input_keep_prob  = self.rnn_para.input_drop, # we have dropouted the output of frame-wise rnn
                                                             output_keep_prob = self.rnn_para.output_drop
                                                            )

            self.rnn_out_01, self.rnn_state_01 = bidirectional_recurrent_layer_output_new(self.fw_cell_01,
                                                                                          self.bw_cell_01,
                                                                                          self.add_time_reshape, #input
                                                                                          self.seq_len,
                                                                                          scope=scope
                                                                                         )
        #Get pooling to feed DNN bl02
        with tf.device('/gpu:0'), tf.variable_scope("pooling_time_dir") as scope:
            self.pool_rnn = tf.reduce_mean(self.rnn_out_01,
                                                axis=[1],
                                                name='pool_add_01'
                                               )


        # ======================================  CNN brand  =================================== #
        # Call CNN and Get CNN output
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01")as scope:
            self.cnn_ins_01 = cnn_bl_conf(self.input_layer_val_01, self.mode)
        with tf.device('/gpu:0'), tf.variable_scope("cnn_02")as scope:
            self.cnn_ins_02 = cnn_bl_conf(self.input_layer_val_02, self.mode)
        with tf.device('/gpu:0'), tf.variable_scope("cnn_03")as scope:
            self.cnn_ins_03 = cnn_bl_conf(self.input_layer_val_03, self.mode)

        self.cnn_ins_01_output = self.cnn_ins_01.final_output
        self.cnn_ins_02_output = self.cnn_ins_02.final_output
        self.cnn_ins_03_output = self.cnn_ins_03.final_output

        # Call ADD and Get Add output
        self.add_cnn = self.cnn_ins_01_output + self.cnn_ins_02_output + self.cnn_ins_03_output

        # Call DNN and Get DNN output
        self.dnn_bl01_ins_02_cnn = dnn_bl01_conf(self.add_cnn, 256, self.mode)
        self.cnn_brand_01        = self.dnn_bl01_ins_02_cnn.final_output  

        
        #======================================= Merge two brands ==========================#
        self.merge_01 = tf.concat([self.cnn_brand_01, self.pool_rnn], 1)
        self.dnn_bl02_ins_01 = dnn_bl02_conf(self.merge_01, 512, self.mode)

        # ======================================  Output =================================== #
        self.output_layer      = self.dnn_bl02_ins_01.final_output
        self.prob_output_layer = tf.nn.softmax(self.output_layer)

        self.wanted_data = self.merge_01 #Extract this data for post-trained process

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            losses  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)

            # final loss
            self.loss = tf.reduce_mean(losses) + l2_loss    #reduce_sum or reduce_mean

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction    = tf.equal(tf.argmax(self.output_layer,1),    tf.argmax(self.expected_classes,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))
