import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt   #jupyter要matplotlib.pyplot
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
#from pytorch_transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
#如果用pytorch_transformers后面的output_all_encoded_layers=False会报错
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
from pandas import DataFrame
import math

from model_config import MyModel_Config  #从其他文件引入我自己定义的类
from data_init import Data_Init
from tqdm import *


'''模型训练的类
该类需要的类MyModel_Config的tensor_datas,epochs,device参数可以被其他类共享
isTPCIL:如果是执行TPCIL增量学习，模型的输出有些不一样，True表示处理TPCIL形式的输出 ,即模型forward的输出多一个cnn_out层的特征输出在probs前面
'''
class Model_Train(object):
    def __init__(self, isTPCIL=False): #
        #self.epochs = MyModel_Config.epochs #训练几个epochs
        self.device = MyModel_Config.device
        self.isTPCIL = isTPCIL
        
    '''测试集和验证集的精度计算,用于全体验证集或测试集的精度计算
    model：要评估的模型
    datapath：输入字符串如'./data/snips/valid.csv'，表明要测试的验证集或测试集路径
    mode:输入字符串'csv'或'tsv' ，表明要测试的文件格式'''
    def my_eval(self, model, datapath, loss_func, mode, label_to_idx_train):
        device = self.device
        tensor_datas, labels_idx = Data_Init(datapath, 64, mode, 'eval', label_to_idx_train).datas_to_tensors()#输出都是tensor形式

        model = model.to(device)
        model.eval() #eval()将我们的模型置于评估模式，而不是训练模式。在这种情况下，评估模式关闭了训练中使用的dropout正则化。
        accuracy=0
        loss_sum=0
        with torch.no_grad():
            for idx, datas in enumerate(tensor_datas):
                tokens = datas[0].to(device)  #tokens输入到bert里得到[batch_size, seq_len, embedding_size]的embedding
                labels_idx = datas[1].to(device)
                
                if self.isTPCIL == False:
                    probs = model(tokens).squeeze()  #去除掉[batch_size, 1, len(classes)]中的1维度
                elif self.isTPCIL == True:
                    _, probs = model(tokens)  #去除掉[batch_size, 1, len(classes)]中的1维度
                    probs.squeeze()
                loss = loss_func(probs, labels_idx) #"host_softmax" not implemented for 'Long'错误出现，如果预测值和标签写反
                #虽然这里的probs没有经过softmax处理，但也可以用下面的这个argmax公式，因为softmax不会改变原本数值元素的大小排名
                accuracy += (labels_idx == torch.argmax(probs, dim=1)).sum()  #计算预测标签和真实标签相等的数量
                loss_sum+=loss.item() #计算所有样本/batch的loss
                
                last_size = len(datas[1])  #用于保存最后一个batch有多少数据

        accuracy = accuracy / (idx*tensor_datas.batch_size + last_size)
        accuracy = accuracy.item()
        model.train()#从eval模式回到train模式

        return accuracy, loss_sum
    
    '''由于增量学习要求对相应的增量类和原始类数据进行精度的计算，所以如果直接输入验证集路径进去，会导致计算所有类精度，所以这里输入变为直接输入数据
    model:要进行精度计算的模型
tensor_datas:验证集/测试集的经过Dataloader封装的数据
loss_function:用于计算验证集/测试集损失'''

    def eval_for_incremental(self, model, tensor_datas, loss_function):
        device = self.device
        model = model.to(device)

        accuracy=0
        loss_sum=0
    
        model.eval() #关闭模型dropout
        with torch.no_grad():
            idx = 0
            for idx, datas in enumerate(tensor_datas):
                tokens = datas[0].to(device)  #tokens输入到bert里得到[batch_size, seq_len, embedding_size]的embedding
                labels_idx = datas[1].to(device)
                
                if self.isTPCIL == False:
                    probs = model(tokens).squeeze()  #去除掉[batch_size, 1, len(classes)]中的1维度
                elif self.isTPCIL == True:
                    _, probs = model(tokens)  #去除掉[batch_size, 1, len(classes)]中的1维度
                    probs.squeeze()
                loss = loss_function(probs, labels_idx) #"host_softmax" not implemented for 'Long'错误出现，如果预测值和标签写反
            
                accuracy += (labels_idx == torch.argmax(probs, dim=1)).sum()  #计算预测标签和真实标签相等的数量
                loss_sum+=loss.item() #计算所有样本/batch的loss
                
                last_size = len(datas[1])  #用于保存最后一个batch有多少数据

        accuracy = float(accuracy) / (idx*tensor_datas.batch_size + last_size)

        model.train()#从eval模式回到train模式

        return accuracy, loss_sum
#用法：
#eval_for_incremental(model, tensor_datas, loss_function),用法在incremental_learning文件的类中

    def eval_for_embeddingKD(self, teacher_embed_model, student_revise, tensor_datas, loss_function):
        device = self.device
        teacher_embed_model = student_revise.to(device)
        student_revise = student_revise.to(device)

        accuracy=0
        loss_sum=0
    
        teacher_embed_model.eval() #关闭模型dropout
        student_revise.eval()
        with torch.no_grad():
            idx = 0
            for idx, datas in enumerate(tensor_datas):
                tokens = datas[0].to(device)  #tokens输入到bert里得到[batch_size, seq_len, embedding_size]的embedding
                labels_idx = datas[1].to(device)

                teacher_embed = teacher_embed_model(tokens)
                probs = student_revise(teacher_embed)

                loss = loss_function(probs, labels_idx) #"host_softmax" not implemented for 'Long'错误出现，如果预测值和标签写反
            
                accuracy += (labels_idx == torch.argmax(probs, dim=1)).sum()  #计算预测标签和真实标签相等的数量
                loss_sum+=loss.item() #计算所有样本/batch的loss
                
                last_size = len(datas[1])  #用于保存最后一个batch有多少数据

        accuracy = float(accuracy) / (idx*tensor_datas.batch_size + last_size)

        student_revise.train()#从eval模式回到train模式

        return accuracy, loss_sum


    '''参数：
    model：训练模型
    loss_func:损失函数
    optimizer:优化器
    epochs:迭代次数
    tensor_datas:要输入的Dataloader封装的数据，默认为MyModel_Config里面的数据
    datapath_eval: 如果等于'none'说明不对验证集或测试集进行每个batch训练后的精度和损失计算；如果等于验证集或测试集路径，则进行计算
    eval_mode：验证集或测试集的格式，为'csv'或'tsv'.
    label_to_idx_train:训练集的标签字典，只有当datapath_eval不为none时候才设置初值'''
    def my_train(self, model, loss_func, optimizer, epochs, tensor_datas, datapath_eval='none', eval_mode='csv', label_to_idx_train={}): #增加了需要自己输入的epochs
        device = self.device
        #epochs = self.epochs
        model.train()

        model = model.to(device)
        losses = [] #存放所有样本一个epoch的损失
        accuracies = []
        iter = [] #用于绘图的横坐标

        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)#每一轮epoch学习率递减
        for epoch in tqdm(range(epochs)):
            
            '''对每个batch的训练'''
            for idx, datas in enumerate(tensor_datas):  #idx表示第几个batch，datas为[batch_size, tokens, label]的数据
                tokens = datas[0].to(device)
                labels = datas[1].to(device)
                #labels_one_hot = datas[1].to(device)  #one-hot形式标签，用于损失计算，[batch_size, labels_nums]
                #labels = torch.topk(labels_one_hot, 1)[1].view(-1,1)   #要计算精度，就需要非one-hot形式的标签，转化为(batch_size,1)形式的标签
        
                optimizer.zero_grad() #新batch训练时将梯度归0，防止梯度累积
                if self.isTPCIL == False:
                    probs = model(tokens).squeeze()  #去除掉[batch_size, 1, len(classes)]中的1维度
                elif self.isTPCIL == True:
                    _, probs = model(tokens)  #去除掉[batch_size, 1, len(classes)]中的1维度
                    probs.squeeze()
                loss = loss_func(probs, labels) #"host_softmax" not implemented for 'Long'错误出现，如果预测值和标签写反
                loss.backward()
                optimizer.step()
                #scheduler.step()#学习率递减
            accuracy_train, loss_sum = self.eval_for_incremental(model, tensor_datas, loss_func)
    
            if datapath_eval != 'none': 
                if label_to_idx_train == {}:
                    raise ValueError("要输出测试集精度模式下需要输入训练集对应的标签字典")
                accuracy_eval, loss_eval = self.my_eval(model, datapath_eval, loss_func, eval_mode, label_to_idx_train)
                print('第'+str(epoch)+'的验证集失为：'+str(loss_eval))
                print('第'+str(epoch)+'的验证集精度为：'+str(accuracy_eval))
            
            accuracies.append(accuracy_train) #accuracy上的数据在cuda上，需要放到cpu上才能作图，而loss.item()已经加到cpu上了
            losses.append(loss_sum)
            iter.append(epoch)
            #print("the loss of  training data "+ str(epoch) + "  is-----------" + str(loss_sum))
            #print("the accuracy of training data   "+ str(epoch) + "  is-----------" + str(accuracy_train))
    
        #plt.figure(1)
        #plt.title("loss of epoch per————"+str(loss_func)+ ","+ str(epochs)+ "epochs")
        #plt.xlabel("loss per epoch")
        #plt.ylabel("LOSS")
        #plt.plot(iter, losses)

        #plt.figure(2)
        #plt.title("accuracy of epoch per————"+str(accuracy_train)+ ","+ str(epochs)+ "epochs")
        #plt.xlabel("accuracy per epoch")
        #plt.ylabel("ACCURACY")
        #plt.plot(iter, accuracies)

        #plt.show()
        return accuracies, losses
    
    '''使用方法，先定义模型，损失函数和优化器，然后实例化Model_Train进行训练，'''
    #bert_lstm = Bert_LSTM(MyModel_Config(atis_labels))
    #loss_fuction = nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(bert_lstm.parameters())
    '''进行训练'''
    #train_lstm = Model_Train().my_train(bert_lstm, loss_fuction, optimizer, epochs, tensor_datas) #普通训练，只输出训练集的损失和精度

    '''单独使用eval用法'''
    #snips_train_tensor_datas, snips_labels, snips_label_idx = Data_Init('./data/snips/train.csv', 8, 'csv', 'train').datas_to_tensors()
    #accuracy = Model_Train().my_eval(bert_lstm, './data/snips/valid.csv', loss_func, 'csv', snips_label_idx)
    #loss_fun是定义好的损失函数

    '''如果要在训练时同时返回测试集和验证集和eval值，那么就
    tensor_datas为Dataloader封装的用于训练的数据'''
    #train_lstm = Model_Train().my_train(bert_lstm, loss_fuction, optimizer, epochs, tensor_datas, '验证集/训练集路径', 'csv/tsv(验证集或测试集格式)', snips_label_idx)
    #snips_label_idx:'train'初始化输出的训练集标签字典