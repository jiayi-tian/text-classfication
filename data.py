# coding: UTF-8

import os
import pickle as pkl
import argparse
import torch
import numpy as np
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def read_txt(class_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    files = os.listdir(class_path)
    for file in files:
        txt_path = os.path.join(class_path, file)
        txt = open(txt_path)
        x = txt.read()
        x = x.strip()
        content = x.split('\t')[0]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    # print(vocab_list)
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    # print(vocab_dic)
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    # print(vocab_dic)
    folder = os.path.basename(class_path)
    lables = {
        '娱乐': 0,
        '游戏': 1,
        '星座': 2,
        '体育': 3,
        '时政': 4,
        '时尚': 5,
        '社会': 6,
        '科技': 7,
        '教育': 8,
        '家居': 9,
        '股票': 10,
        '房产': 11,
        '彩票': 12,
        '财经': 13,
    }
    lable = lables[folder]
    print(lable)

    return vocab_dic, lable

def build_dataset(path, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    # if os.path.exists("./data/vocab_new.pkl"):
    #     vocab = pkl.load(open("./data/vocab_new.pkl", 'rb'))
    # else:
    vocab, lable = read_txt(path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump((vocab, lable), open("./data/vocab_new.pkl", 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        files = os.listdir(path)
        for file in files:
            txt_path = os.path.join(path, file)
            txt = open(txt_path)
            x = txt.read()
            x = x.strip()
            content = x.split('\t')[0]
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(lable), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(path, pad_size=32)
    # print(train)
    return vocab, train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, device):
    iter = DatasetIterater(dataset, batch_size=128, device=device)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



if __name__ == "__main__":
    tokenizer = lambda x: [y for y in x]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    vocab, lable = read_txt("./娱乐", tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    args = parser.parse_args()
    emb_dim = 300
    all_path = "./THUCNews"
    filename_trimmed_dir = "./data/embedding_SougouNews"
    pretrain_dir = "./data/sgns.sogou.char"
    folders = os.listdir(all_path)
    for folder in folders:
        folder_path = os.path.join(all_path, folder)
        vocab, train = build_dataset(folder_path, args.word)
        embeddings = np.random.rand(len(vocab), emb_dim)
        f = open(pretrain_dir, "r", encoding='UTF-8')
        for i, line in enumerate(f.readlines()):
            # if i == 0:  # 若第一行是标题，则跳过
            #     continue
            lin = line.strip().split(" ")
            if lin[0] in vocab:
                idx = vocab[lin[0]]
                emb = [float(x) for x in lin[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        f.close()
        np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)


