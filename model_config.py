import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset, TensorDataset

'''调整模型参数和模型训练的类
包含bert模型的embedding size，路径
CNN模型的卷积核数量，卷积尺寸以及步长
LSTM模型的hidden_size,dropout，hidden layers
训练过程的epochs，device
Dataloader封装的数据'''
class MyModel_Config(object):  #模型命名不能与官方的重名，如Model_Config
    '''
    配置参数
    labels: 之前得到的类别list，里面包含数据文件的所有类别 ,(对应参数labels)
    '''
    def __init__(self, labels):  #初始化该类的时候输入dataaset,tensor_datas,labels
        #self.model_name='Bert CNN Model'
        
        '''bert相关设置'''
        self.bert_path='./bert-pretrained'
        self.bertmini_path='./bert_mini'
        self.hidden_size=768  # Bert模型 token的embedding维度 = Bert模型后接自定义分类器（单隐层全连接网络）的输入维度
        self.hiddne_size_mini=256
        
        '''数据相关设置''' 
        #类中self.是只有本类中能使用的变量，但是类名.的变量是可以其他类访问的  （Module除外）

        # 类别数
        self.num_classes = len(labels)
        
        '''训练相关设置'''
        # 整体训练次数
        MyModel_Config.epochs=5
        # 配置使用检测GPU
        MyModel_Config.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        # MyModel_Config.tensor_datas = tensor_datas #DataLoader封装好的数据和标签  

        self.dropout = 0.5 #LSTM和CNN模型的dropout

        '''CNN 相关设置'''
        self.num_filters=256 #每个filter_size的CNN的卷积核数量，即用来和sample做卷积的那个矩阵，其个数有256个
        self.filter_size=(2,3,4) #有三个卷积核，其高度分别为2，3，4(设置为3,4,5有利于多类数据集的学习)

        '''LSTM相关设置'''
        self.LSTM_hidden_dim = 256
        self.LSTM_layers = 3
        self.bidirectional=True
        self.bias = True
        

        '''XLnet设置'''
        self.xlnet_hidden_dim = 256
        self.xlnet_n_layers = 2
        self.xlnet_bidirectional = True
        self.xlnet_dropout=0.1
