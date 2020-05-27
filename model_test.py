# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:50:18 2020

@author: 50232
"""

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import sys
import json
import pandas as pd
import random
from datasetProcessing import DataTest
from model_train import vedio2text


learning_rate = 0.001
num_epoches = 100
batch_size = 100
num_display_steps = 15
num_saver_epoches = 1
save_dir = 'save_model/'
log_dir = 'logs/'
output_filename = '/final_output.txt'
data_dir = sys.argv[1]
test_dir = sys.argv[2]
#data_dir = '/home/lingyis/DLhw2/MLDS_hw2_1_data'
#test_dir = '/home/lingyis/DLhw2/MLDS_hw2_1_data/testing_data'

special_tokens  = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
phases = {'train': 0, 'val': 1, 'test': 2}
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)
n_inputs = 4096
n_hidden = 256
val_batch_size = 100
n_frames = 80
max_caption_len = 50
forget_bias_red = 1.0
forget_bias_gre = 1.0
dropout_prob = 0.5

def test_print(pred, idx2word, batch_size, id_batch):
    
    seq = []
    for i in range(0, batch_size):
        eos_pred = max_caption_len - 1
        for j in range(0, max_caption_len):
                if pred[i][j] == special_tokens['<EOS>']:
                    eos_pred = j
                    break
        pre = list( map (lambda x: idx2word[x] , pred[i][0:eos_pred])  )
        print('\nid: ' + str(id_batch[i]) + '\nlen: ' + str(eos_pred) + '\nprediction: ' + str(pre))
        pre_no_eos = list( map (lambda x: idx2word[x] , pred[i][0:(eos_pred)])  )
        sen = ' '.join([w for w in pre_no_eos])
        seq.append(sen)
    return seq

def test():

    datasetTest = DataTest(data_dir, test_dir, batch_size)
    datasetTest.build_test_data_obj_list()
    vocab_num = datasetTest.load_tokenizer()

    test_graph = tf.Graph()

    with test_graph.as_default():
        feat = tf.placeholder(tf.float32, [None, n_frames, n_inputs])
        model = vedio2text(vocab_num=vocab_num)
        logits, _, _ = model.build_model(feat, phase=phases['test'])
        dec_pred = model.inference(logits)

        model.set_saver(tf.train.Saver(max_to_keep=3))
    sess = tf.Session(graph=test_graph)
    saver_path = save_dir
    print('model path: ' + saver_path)
    latest_checkpoint = tf.train.latest_checkpoint(saver_path)
    
    model.saver.restore(sess, latest_checkpoint)
    txt = open(output_filename, 'w')

    num_steps = int( datasetTest.batch_max_size / batch_size)
    for i in range(0, num_steps):
        data_batch, id_batch = datasetTest.next_batch()
        p = sess.run(dec_pred, feed_dict={feat: data_batch})
        seq = test_print(p, datasetTest.idx_to_word, batch_size, id_batch)

        for j in range(0, batch_size):
            txt.write(id_batch[j] + "," + seq[j] + "\n")
    
    print('\n Testing finished.')
    txt.close()

def main(_):
    test()
    
if __name__ == '__main__':
    tf.app.run(main=main)