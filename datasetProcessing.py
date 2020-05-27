# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from datasetBase import DatasetBase, DataObject


class DataTest(DatasetBase):
    def __init__(self, data_dir, test_dir, batch_size = 20):
        super().__init__(data_dir, batch_size)
        self.test_dir = test_dir
        self.feat_dir = self.test_dir + '/feat/'
        self.batch_size = batch_size
        self.id_txt = '/id.txt'

    def load_tokenizer(self):

        with open('word_index.pkl', 'rb') as handle:
            self.word_index = pickle.load(handle)
        with open('idx_to_word.pkl', 'rb') as handle:
            self.idx_to_word = pickle.load(handle)
        with open('word_counts.pkl', 'rb') as handle:
            self.word_counts = pickle.load(handle)

        self.vocab_num = len(self.word_counts) + 4
        return self.vocab_num

    def build_test_data_obj_list(self):

        txt = open(self.test_dir + self.id_txt, 'r')
        print('load txt: ' + self.test_dir + self.id_txt)
        max_size = 0

        for line in txt.readlines():

            myid = line.split('\n')[0]
            path = self.feat_dir + myid + '.npy'
            mydat = np.load(path)
           
            obj = DataObject(path, myid)
            self.dat_dict[myid] = mydat
            max_size += 1
            self.data_obj_list.append(obj)
        
        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = max_size

    def next_batch(self): 
        
        current_index = self.batch_index
        max_size = self.batch_max_size
        if current_index + self.batch_size <= max_size:
            dat_list = self.data_obj_list[current_index:(current_index + self.batch_size)]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            dat_list = np.append(self.data_obj_list[current_index:max_size], self.data_obj_list[0: right])
            self.batch_index = right
        
        img_batch = []
        id_batch = []
        for d in dat_list:
            img_batch.append(self.dat_dict[d.myid])
            id_batch.append(d.myid)
            
        return np.array(img_batch), np.array(id_batch)
    
filters = '`","?!/.()'
special_tokens_to_word = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
max_caption_len = 50
random.seed(0)
np.random.seed(0)

class DataTrain(DatasetBase):

    def __init__(self, data_dir, batch_size):
        super().__init__(data_dir, batch_size)
        self.feat_dir = self.data_dir + '/training_data/feat/'
        self.json_filename = '/training_label.json'
        self.corpus_dir = self.data_dir
        self.perm = None # permutation numpy array

    def prep_token_list(self):
        corpus_path = self.corpus_dir + self.json_filename
        train_file = pd.read_json(corpus_path)
        total_list = []
        for i in range(0, len(train_file['caption'])):
            str_list = train_file['caption'][i]
            for j in range(0, len(str_list)):
                total_list.append(str_list[j])
        return total_list

    def dump_tokenizer(self):
        total_list = self.prep_token_list()

        tokenizer = Tokenizer(filters=filters, lower=True, split=" ")
        tokenizer.fit_on_texts(total_list)

        for tok in tokenizer.word_counts.items():
            if tok[1] >= self.word_min_counts_threshold:
                self.word_counts[tok[0]] = tok[1]

        self.vocab_num = len(self.word_counts) + 4 # init vocab_num, must add 4 special tokens!!

        for i in range(0, 4):
            tok = special_tokens_to_word[i]
            self.word_index[tok] = i
            self.idx_to_word[i] = tok

        cnt = 0
        for tok in tokenizer.word_index.items():
            if tok[0] in self.word_counts:
                self.word_index[tok[0]] = cnt + 4
                self.idx_to_word[cnt + 4] = tok[0]
                cnt += 1

        assert len(self.word_index) == self.vocab_num # yes! they are equal

        with open('word_index.pkl', 'wb') as handle:
            pickle.dump(self.word_index, handle)
        with open('idx_to_word.pkl', 'wb') as handle:
            pickle.dump(self.idx_to_word, handle)
        with open('word_counts.pkl', 'wb') as handle:
            pickle.dump(self.word_counts, handle)
        return self.vocab_num # for embedding 

    def build_train_data_obj_list(self):
        corpus_path = self.corpus_dir + self.json_filename

        data_file = pd.read_json(corpus_path)
        max_size = 0
        for i in range(0, len(data_file['caption'])):

            myid = data_file['id'][i]
            path = self.feat_dir + myid + '.npy'
            mydat = np.load(path)
            str_list = data_file['caption'][i]
            self.dat_dict[myid] = mydat
            #repeat = {}

            for j in range(0, len(str_list)):
                tmp_list = []
                cap_len_list = []

                seq = text_to_word_sequence(str_list[j], filters=filters, lower=True, split=" ")

                tmp_list.append(seq)
                cap_len_list.append(len(seq) + 1) # added <EOS> !!
                obj = DataObject(path, myid, tmp_list, cap_len_list)
                max_size += 1
                self.data_obj_list.append(obj)

        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = max_size
        self.perm = np.arange( self.batch_max_size, dtype=np.int )
        self.shuffle_perm()

    def shuffle_perm(self):
        np.random.shuffle( self.perm )

    def next_batch(self): 
        
        # 1. sequential chosen
        current_index = self.batch_index
        max_size = self.batch_max_size
        if current_index + self.batch_size <= max_size:
            dat_list = self.data_obj_list[self.perm[current_index:(current_index + self.batch_size)]]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            dat_list = np.append(self.data_obj_list[self.perm[current_index:max_size]], 
                    self.data_obj_list[self.perm[0: right]])
            self.batch_index = right

        img_batch = []
        cap_batch = []
        id_batch = []
        cap_len = []
        for d in dat_list:
            img_batch.append(self.dat_dict[d.myid])
            id_batch.append(d.myid)
            cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list)
            cap = np.array(cap)
            cap_batch.append(cap)
            cap_len.append(l)
        cap_batch = self.captions_to_padded_sequences(cap_batch)

        return np.array(img_batch), np.array(cap_batch), np.array(cap_len), np.array(id_batch)

    def schedule_sampling(self, sampling_prob, cap_len_batch):

        sampling = np.ones(max_caption_len, dtype = bool)
        for l in range(max_caption_len):
            if np.random.uniform(0,1,1) < sampling_prob:
                sampling[l] = True
            else:
                sampling[l] = False
         
        sampling[0] = True
        return sampling


class DataVal(DatasetBase):
    def __init__(self, data_dir, batch_size):
        super().__init__(data_dir, batch_size)
        self.feat_dir = self.data_dir + '/testing_data/feat/'
        self.json_filename = '/testing_label.json'
        self.corpus_dir = self.data_dir    

    def load_tokenizer(self):
        # should be put in same folder!
        with open('word_index.pkl', 'rb') as handle:
            self.word_index = pickle.load(handle)
        with open('idx_to_word.pkl', 'rb') as handle:
            self.idx_to_word = pickle.load(handle)
        with open('word_counts.pkl', 'rb') as handle:
            self.word_counts = pickle.load(handle)

        self.vocab_num = len(self.word_counts) + 4
        return self.vocab_num

    def build_val_data_obj_list(self):
        
        corpus_path = self.corpus_dir + self.json_filename
        data_file = pd.read_json(corpus_path)
        max_size = 0
        for i in range(0, len(data_file['caption'])):

            myid = data_file['id'][i]
            path = self.feat_dir + myid + '.npy'
            mydat = np.load(path)
            str_list = data_file['caption'][i]
           
            tmp_list = []
            cap_len_list = [] 
            for j in range(0, len(str_list)):
                seq = text_to_word_sequence(str_list[j], filters=filters, 
                    lower=True, split=" ")
                tmp_list.append(seq)
                cap_len_list.append(len(seq) + 1) # added <EOS>
            obj = DataObject(path, myid, tmp_list, cap_len_list)
            self.dat_dict[myid] = mydat
            max_size += 1
            self.data_obj_list.append(obj)
        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = max_size

    def next_batch(self): 

        current_index = self.batch_index
        max_size = self.batch_max_size
        if current_index + self.batch_size <= max_size:
            dat_list = self.data_obj_list[current_index:(current_index + self.batch_size)]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            dat_list = np.append(self.data_obj_list[current_index:max_size], self.data_obj_list[0: right])
            self.batch_index = right      
        img_batch = []
        cap_batch = []
        id_batch = []
        cap_len = []
        for d in dat_list:
            img_batch.append(self.dat_dict[d.myid])
            id_batch.append(d.myid)
            cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list) # randomly pick one
            cap = np.array(cap)
            cap_batch.append(cap)
            cap_len.append(l)
        cap_batch = self.captions_to_padded_sequences(cap_batch)
        return np.array(img_batch), np.array(cap_batch), np.array(cap_len), np.array(id_batch)