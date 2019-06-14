# -*- coding: utf-8 -*-

import time
import sys
import argparse
from flask import Flask, request
import random
import json
import copy
import torch
import gc
import _pickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.metric import get_ner_fmeasure
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

app = Flask(__name__)

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))

        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def inference(data, model):
    instances = data.raw_Ids
    instance = instances[:1]
    model.eval()
    gaz_list,batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
    tag_seq = model(gaz_list,batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
    pred_label, _ = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
    return pred_label

def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data

def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    print(words)
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    print(word_seq_lengths)
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile = volatile_flag).byte()

    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.LongTensor([1]*int(seqlen.squeeze().item()))

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    ## not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    ## keep the gaz_list in orignial order

    gaz_list = [ gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def get_entity_from_labels_BIO_schema(text):
    res = []
    beg = -1
    cur = 0
    tpe = ''

    for i in range(len(text)):
        if cur == 0 and text[i] == 'O':
            continue
        if text[i][0] == 'I':
            continue
        elif text[i][0] == 'B':
            beg = i
            cur = 1
            tpe = text[i][2:]
        else:
            cur = 0
            res.append((beg, i-1, tpe))
            tpe = ''

    res = json.dumps(res)
    return res

def data_transform(text):
    return [y + ' O' for y in list(text)] + [' ']

class Tagger():
    def __init__(self, model_dir, dset_dir, gpu, seg):
        self.model_dir = model_dir
        self.dset_dir = dset_dir
        self.data = load_data_setting(dset_dir)
        self.data.HP_gpu = gpu
        self.model = SeqModel(self.data)
        self.model.load_state_dict(torch.load(self.model_dir))

    def change_inlines(self, text):
        self.data.inference_single_with_gaz(text)

    def load_model_inference(self, seg=True):
        #self.model = SeqModel(self.data)
        return inference(self.data, self.model)

tg = None

@app.route('/api/ner/cner', methods=['POST', 'GET'])
def online_inference():
    if request.method == 'GET':
        in_lines = request.args.get('text')
    else:
        in_lines = request.form['text']
        #in_lines = '北京人民需要共产党的领导'
    in_lines = [x if x != '\n' else '。' for x in in_lines]
    in_lines = data_transform(in_lines)
    #print(in_lines)
    tg.change_inlines(in_lines)
    #print('e',data.raw_Ids)
    decode_results = tg.load_model_inference() #model_dir, data, 'raw', gpu, seg)
    print(decode_results)
    return get_entity_from_labels_BIO_schema(decode_results[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding',  help='Embedding for words', default='None')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    args = parser.parse_args()

    model_dir = args.loadmodel
    dset_dir = args.savedset

    print('model_dir {}, dset_dir {}'.format(model_dir, dset_dir))

    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()

    sys.stdout.flush()

    tg = Tagger(model_dir, dset_dir, gpu, seg)
    print('begin to run.....')
    app.run('0.0.0.0', port=4444)
