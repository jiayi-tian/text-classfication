# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:17:33 2019

@author: 12642
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
#import time
#from utils import get_time_dif
#from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
num_epochs = 10 
learning_rate = 1e-3
batch_size = 128 #？？？
#require_improvement = 1000 
#save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
#class_list = [x.strip() for x in open(
            #dataset + '/data/class.txt').readlines()] 
def train(model, train_iter, test_iter):
    #start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    #dev_best_loss = float('inf')
    #last_improve = 0  # 记录上次验证集loss下降的batch数
    #flag = False  # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()#把梯度置零，也就是把loss关于weight的导数变成0
            loss = F.cross_entropy(outputs, labels)#求出loss的值
            loss.backward()#反向传播求梯度（偏导）
            optimizer.step()#更新所有参数
            if total_batch % 100 == 0:#？？？
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)#所有分类正确的百分比，默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
                #dev_acc, dev_loss = evaluate(model, dev_iter)#验证集多次使用，不断调参
                '''if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(),'train.pth')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                #time_dif = get_time_dif(start_time)'''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, {6}'
                print(msg.format(total_batch, loss.item(), train_acc, improve))
                #writer.add_scalar("loss/train", loss.item(), total_batch)
                #writer.add_scalar("loss/dev", dev_loss, total_batch)
                #writer.add_scalar("acc/train", train_acc, total_batch)
                #writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            '''if total_batch - last_improve >require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break'''
   #writer.close()
    test(model, test_iter)


def test(model, test_iter):
    # test
    model.load_state_dict(torch.load('train.pth'))
    #model.eval()
    #start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    #time_dif = get_time_dif(start_time)
    #print("Time usage:", time_dif)


def evaluate(model, data_iter, test=False):
    #model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)