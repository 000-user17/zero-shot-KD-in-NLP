from this import d
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random


'''把训练数据tokenize后做padding，然后把token和对应的label封装好到DataLoader中，形成[batch_size, sequence_len, embedidng_size]的形式
datas_path: 训练文件所在的路径，是字符串，如  './data/atis/train.csv'
batch_size: 训练数据的一个batch又多少样本数据
data_type:输入为字符串'csv'或'tsv',处理不同格式的数据
mode: 输入字符串'train'或者'eval'，其中train模式输出Dataloader封装好的训练数据以及所有的类别，
eval模式用于验证集和测试集，输出经过bert tokenize后的token idx tensor（用于直接让模型进行forward），以及每个token对应的标签tensor（用于精度计算）
label_to_idx_train:eval模式下，测试集的标签字典需要和训练集生成的标签字典对应，因此输入为训练集的标签字典
'''
class Data_Init(object):
    def __init__(self, datas_path, batch_size, data_type, mode='train', label_to_idx_train={}):
        self.datas_path = datas_path
        self.batch_size = batch_size
        self.data_type = data_type
        self.mode = mode
        self.label_to_idx_train = label_to_idx_train
    '''
    注意：注意这里的csv文件的标签栏默认为 'label',如有不同需更改，句子标签默认为'text',否则会无法读取数据
    '''
    def datas_to_tensors(self):
        '''从文件读取数据'''
        datas_path = self.datas_path
        batch_size = self.batch_size
        data_type = self.data_type 
        mode = self.mode

        '''不同格式文件的提取'''
        if data_type == 'csv':  
            datas = pd.read_csv(datas_path)
        elif data_type == 'tsv':
            datas = pd.read_csv(datas_path, sep='\t', header=0)

        datas_size = datas.shape[0]
        #print('数据大小形状为:'+str(datas.shape))

        '''创建类别字典
        labels:保存训练数据中所有的类别
        label_to_idx:类别的数据字典
        '''

        labels = [] #保存atis train.csv中一共有多少类
        for line in range(datas_size):
            if datas['label'][line] not in labels:          #注意这里的csv文件的标签栏默认为 'label',如有不同需更改
                labels.append(datas['label'][line])
        #print('训练数据所有的label：'+str(labels))
        
        label_to_idx={} #label字典
        if mode == 'train':
            for l in range(len(labels)):
                label_to_idx[labels[l]] = l
            #print('label字典：'+str(label_to_idx))
        
        elif mode == 'eval':
            if self.label_to_idx_train == {}:
                raise ValueError("eval模式需要输入训练集的标签字典来对应测试集生成的标签idx")
            else:
                label_to_idx = self.label_to_idx_train

        '''准备数据，先将atis中的数据用bert进行tokenize'''
        train_text = []#存放训练数据中未进行tokenize的text
        labels_idx=[] #存放训练数据每个text对应的label的index （int形式，不是one-hot向量）

        for l in range(datas_size):
            train_text.append('[CLS] '+datas['text'][l]+' [SEP]')  #这里记得[CLS]加空格，空格加[SEP]否则会不能成功分词
            label = label_to_idx[datas['label'][l]] #label为int形式的标签

            labels_idx.append(label)  
            #labels_idx.append([label]) ##int形式的标签必须要这样[]的形式才能转化为one-hot

        #labels_vec = torch.zeros(len(labels_idx), len(label_to_idx)).scatter_(1, labels_idx, 1)
        #将int形式的标签转化为train数据个数个，长度为标签数量的one-hot向量 

        tokenizer = BertTokenizer.from_pretrained('./bert-pretrained') #必须要./表示当前文件夹的某个文件
        train_data_tokens = [] #用于存放tokenize后的训练数据tokens

        for l in range(datas_size):
            tokens = tokenizer.tokenize(train_text[l])
            train_data_tokens.append(tokens)

        '''得到每个token的index值，并做padding,也可以用一个mask的list，如果是[pad]符号则为0，其他token为1，与原始的token序列连接'''
        max_len=0   #训练数据中最大的token长度进行padding时候用
        for i in range(datas_size):
            max_len = max(max_len, len(train_data_tokens[i]))
        #print('训练数据中最长的token长度为：'+str(max_len))

        train_tokens_idx=[]
        for i in range(datas_size):
            token_idx = tokenizer.convert_tokens_to_ids(train_data_tokens[i])
            while len(token_idx) < max_len:
                token_idx.append(0)                  #bert的[pad]对应的index为0，所以添加0做padding
            train_tokens_idx.append(token_idx)

        '''随机打乱训练集和测试集数据，防止学习到数据本身的顺序特征'''
        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(train_tokens_idx)
        random.seed(randnum)
        random.shuffle(labels_idx)  #打乱训练集数据,这里注意必须打乱list类型的数据集，torch类型会导致重复

        labels_idx = torch.tensor(labels_idx)
        '''DataLoader的使用，先用Dataset创建然后传入Dataloader'''
        tensor_datasets = TensorDataset(torch.tensor(train_tokens_idx), labels_idx)
        #atis_labels_vec已经是one-hot形式了，atis_train_tokens_idx是list形式
        train_tensor_datas = DataLoader(tensor_datasets, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        #batch中有128个样本,dro_last表示最后如果不能凑齐一个batch是否丢弃数据,batch_size过大可能会导致cuda _outif_memory
        #这里貌似封装了后才能正常计算loss，否则会报错

        if mode == 'train':
            return train_tensor_datas, labels, label_to_idx
            #返回DataLoader封装好的训练数据，形为[batch_size, seq_len(tokens), labels_one_hot]； 以及一共有多少个标签类别，用于传输给model_config来确定softmax层神经元个数
        
        elif mode == 'eval':
            return train_tensor_datas, labels_idx #labels_idx已经转换成tensor了
        #返回封装好的数据和labels index和其对应的labels的index
    '''这里'test'模式return的东西和train不一样，因为为了方便计算，'test'返回的是没有封装的token tensor和label的tensor index值
    但返回的都是tensor形式'''


'''训练集数据封装使用方法'''
#snips_train_tensor_datas, snips_labels, snips_label_idx = Data_Init('./data/snips/train.csv', 8, 'csv', 'train').datas_to_tensors()
#snips_train_tensor_datas

'''测试集和验证集数据封装使用方法'''
#oos_train_tensor_datas, labels_idx = Data_Init('./data/oos/train.tsv', 8, 'tsv', 'eval', snips_label_idx).datas_to_tensors()
#oos_train_tensor_datas



'''增量学习数据初始化类'''
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
from pandas import DataFrame


'''用于生成类增量学习要用到的原始类的训练数据，以及增量类的数据
模型现在原始类的数据上进行训练，然后在其他类上分别进行增量学习，例如增量类数目为2，则每次类增量学习就用这新的两个类训练老模型
datas_path：文件路径。如要生成atis train.csv的数据，则要输入 './data/atis/train.csv'
data_type:要读入文件的格式，填'csv'或'tsv'
batch_size:要生成Dataloader文件的batchsize大小
original_class_num:要生成的原始类个数
incremental_class_num：每次的增量类个数
drop_last: 类增量分类时，对增量类，如果分完原始类后的剩下类数目不能整除每次增量类个数，是否舍弃不能整除的类别部分（True表示舍弃）
data_num:对数据集大小进行划分，按照数据集的顺序，从上往下选取data_num个数据作为一个类别的数据
mode:为了保证训练集和测试集，验证集划分的original class和incremental class顺序一致，输入为'train'或'eval',train模式生成训练数据集的labels和datas，eval模式也利用训练集的labels顺序生成其labels和datas
labels：eval模式要输入训练集的labels保证label划分的和训练数据划分的一致性
label_to_idx: eval模式要输入训练集的labels保证label字典的一致性，以确保训练集和测试集同类的标签索引一致

注：三个函数的思路  prepare_for_incremental -> My_DataLoader and Joint_incremental
prepare_for_incremetal的中间步骤是My_DataLoader，生成Dataloader返回给prepare_for_incremetal
Joint_incremental是My_DataLoader的中间过程，利用init中的self变量list来存放用于JOint模型训练的数据
'''
class class_incremental(object):
    '''定义全局变量，在该类中的程序运行中会保存，如果是在init定义self，其内容只会在调用其的程序中保存
    使用时class_incremental.Joint_incremental_tokens
    测试代码
    '''

    def __init__(self, datas_path, data_type, batch_size, original_class_num, incremental_class_num, drop_last, data_num, mode = 'train', labels=['none'], label_to_idx = ['none']):
        self.datas_path = datas_path
        self.data_type = data_type
        self.original_class_num = original_class_num
        self.incremental_class_num = incremental_class_num
        self.batch_size = batch_size
        self.mode = mode
        self.labels = labels
        self.drop_lase = drop_last
        self.data_num = data_num
        self.label_to_idx = label_to_idx

        self.Joint_incremental_labels = []
        self.Joint_incremental_tokens = []
        self.Joint_incremental_datasets = []

    '''中间函数，用于prepare_for_incremental的调用，代码从data_init.py搬来的，去除了读取数据部分，避免再更改data_init函数，再增加参数
    datas: 本类下prepare_for_incremental()中间过程文件，本质为pandas读取了csv/tsv数据后的数据结构
    max_padding_size:统一同一数据下的要padding的长度，避免原始类和各个增量类之间padding长度不同，导致训练出现问题
    label_to_idx:数据集所有数据生成的标签字典索引'''
    def My_DataLoader(self, datas, max_padding_size, label_to_idx):

        batch_size = self.batch_size #每个batch的大小
        datas_size = datas.shape[0]

        '''准备数据，先将如atis中的数据用bert进行tokenize'''
        train_text = []#存放训练数据中未进行tokenize的text
        labels_idx=[] #存放训练数据每个text对应的label的index （int形式，不是one-hot向量）

        for l in range(datas_size):
            train_text.append('[CLS] '+datas['text'][l]+' [SEP]')
            label = label_to_idx[datas['label'][l]] #label为int形式的标签

            labels_idx.append(label)  #int形式的标签必须要这样[]的形式才能转化为one-hot
        
        self.Joint_incremental_labels += labels_idx   #合并

        labels_idx = torch.tensor(labels_idx)

        tokenizer = BertTokenizer.from_pretrained('./bert-pretrained') #必须要./表示当前文件夹的某个文件
        train_data_tokens = [] #用于存放tokenize后的训练数据tokens

        for l in range(datas_size):
            tokens = tokenizer.tokenize(train_text[l])
            train_data_tokens.append(tokens)
        
        '''padding：给不足max_padding_size长度的token index补0'''
        train_tokens_idx=[]
        for i in range(datas_size):
            token_idx = tokenizer.convert_tokens_to_ids(train_data_tokens[i])
            while len(token_idx) < max_padding_size:
                token_idx.append(0)                  #bert的[pad]对应的index为0，所以添加0做padding
            train_tokens_idx.append(token_idx)

        self.Joint_incremental_tokens += train_tokens_idx  #合并之前的数据
        Joint_incremental_tensor_datasets = TensorDataset(torch.tensor(self.Joint_incremental_tokens), torch.tensor(self.Joint_incremental_labels))
        self.Joint_incremental_datasets.append(Joint_incremental_tensor_datasets) #是一个list，将每轮（包括orgin）增量类的dataset数据存放作为元素

        '''DataLoader的使用，先用Dataset创建然后传入Dataloader'''
        tensor_datasets = TensorDataset(torch.tensor(train_tokens_idx), labels_idx)
        #atis_labels_vec已经是one-hot形式了，atis_train_tokens_idx是list形式
        train_tensor_datas = DataLoader(tensor_datasets, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        #batch中有128个样本,dro_last表示最后如果不能凑齐一个batch是否丢弃数据,batch_size过大可能会导致cuda _outif_memory

        return train_tensor_datas

    '''用于生成用于Joint_incremental学习所用到的数据，即每次增量类来后，要用原始类数据和增量类数据一起训练
    因此，该方法输出一个list，list的每个元素都为原始数据和第k次增量类数据的Dataloader封装的数据
    注意：Joint_incremental_tensor_datas[0]为original data，因此要用到增量类数据要从注意：Joint_incremental_tensor_datas[1]开始
    这里的self.Joint_incremental_datasets相当于类内的全局变量，比较稳定，只要操作时加上self，就可以保存其值'''
    def Joint_incremental(self):

        Joint_incremental_datasets = self.Joint_incremental_datasets
        Joint_incremental_tokens = self.Joint_incremental_tokens
        Joint_incremental_labels = self.Joint_incremental_labels
        batch_size = self.batch_size

        Joint_incremental_tensor_datas = [] #存放每一轮经过DataLoader封装的Joint_incremental_datasets数据作为元素

        for i in range(len(Joint_incremental_datasets)):
            Joint_incremental_datas = DataLoader(Joint_incremental_datasets[i], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
            Joint_incremental_tensor_datas.append(Joint_incremental_datas)

        return Joint_incremental_tensor_datas #一个list，每个元素为每次的老类和第k次增量类的dataloader
 #用法：要先执行prepare_for_incremental函数才能得到Joint_incremental的输出
#incremental = class_incremental('./data/oos/train.tsv', 'tsv', 128, 50, 20, True, 100)
#original_tensor_datas, incremental_tensor_datas_list, original_labels, all_incremental_labels, labels = incremental.prepare_for_incremental()
#Joint_incremental_datasets = incremental.Joint_incremental()



    '''要调用的函数。
    把类增量学习要用到的original class的数据和标签放入DataLoader里，形成[batch_size, sequence_len, label]
    返回值有两个：original_tensor_datas：原始类的Dataloader封装数据
    incremental_tensor_datas_list：一个list，里面的元素为每次Dataloader封装的增量类数据'''
    def prepare_for_incremental(self):
        mode = self.mode
        data_num = self.data_num

        if mode == 'eval':
            if self.labels == ['none']:
                raise Exception("'eval'模式下请输入训练集生成的的labels")    #抛出eval模式没输入训练集labels的提示
            if self.label_to_idx == ['none']:
                 raise Exception("'eval'模式下请输入训练集生成的的label_to_idxs")

        datas_path = self.datas_path #
        data_type = self.data_type
        original_class_num = self.original_class_num
        incremental_class_num = self.incremental_class_num
        drop_last = self.drop_lase

        '''不同格式文件的提取'''
        if data_type == 'csv':  
            datas = pd.read_csv(datas_path)
        elif data_type == 'tsv':
            datas = pd.read_csv(datas_path, sep='\t', header=0)

        datas_size = datas.shape[0]

        '''对所有类创建类别字典
        labels:保存训练数据中所有的类别
        label_to_idx:类别的数据字典
        '''

        labels = [] #保存atis train.csv中一共有多少类
        if mode == 'train': #训练模式需要自己生成labels表
            for line in range(datas_size):
                if datas['label'][line] not in labels:          #注意这里的csv文件的标签栏默认为 'label',如有不同需更改
                    labels.append(datas['label'][line])
            #print('训练数据所有的label：'+str(labels))

        label_to_idx = {}   #label_to_idx顺序和labels顺序一致
        if self.mode == 'train':  #训练数据生成的label字典，也要用于测试集和验证集的label字典
            for l in range(len(labels)):
                label_to_idx[labels[l]] = l
       
        '''测试集和验证集保证和训练集划分label顺序一致'''
        if mode == 'eval':
            labels = self.labels
            label_to_idx = self.label_to_idx
        
        '''统一同一数据集最大的padding_size'''
        train_text = []
        for l in range(datas_size):
            train_text.append('[CLS]'+datas['text'][l]+'[SEP]')
        tokenizer = BertTokenizer.from_pretrained('./bert-pretrained') #必须要./表示当前文件夹的某个文件

        max_padding_size=0  
        for l in range(datas_size):
            tokens = tokenizer.tokenize(train_text[l])
            max_padding_size = max(max_padding_size, len(tokens))


        '''生成原始类数据'''
        original_labels = labels[0:original_class_num]#取到原始基础训练类个数个标签类,取labels从0到original_class_num-1
        original_class_texts = [] #存放原始类的文本
        original_class_labels = []#存放原始类文本对应的标签

        original_datanum_dict = {}  #初始化每个label对应的datanum数量
        for label in original_labels:
            original_datanum_dict[label] = 0 

        for line in range(datas_size):
            if datas['label'][line] in original_labels:

                if original_datanum_dict[datas['label'][line]] >= data_num: #如果某个label的数量超过了设置的data_num，则跳过该label行
                    continue
                original_datanum_dict[datas['label'][line]] += 1
                
                original_class_texts.append(datas['text'][line])
                original_class_labels.append(datas['label'][line])
        '''打乱训练数据集顺序'''
        if mode == 'train':
            randnum = random.randint(0,100)
            random.seed(randnum)
            random.shuffle(original_class_texts)
            random.seed(randnum)
            random.shuffle(original_class_labels)  #打乱训练集数据,这里注意必须打乱list类型的数据集，torch类型会导致重复
                
        '''为原始类生成Dataloader封装好的数据，其内含batch数量个[batch_size, seq_len, labels(one-hot)]形式的数据
        这种类型的数据可以直接用于我写好的train函数'''
        original_dataframe = pd.DataFrame({'text':original_class_texts,'label':original_class_labels}) #将其储存为panda读取csv/tsv文件后的形式
        original_tensor_datas = self.My_DataLoader(original_dataframe, max_padding_size, label_to_idx) 
        
        '''用于生成类增量类数据'''
        all_incremental_labels = [] #元素为每个类增量类的标签的list ,用于搞清楚都用了那些类做类增量
        incremental_tensor_datas_list = [] #list，元素为每次增量类DataLoader封装的数据

        for l in range( int((len(labels) - original_class_num) / incremental_class_num) ): #记住要转化为int才能向上取整，先获得增量学习次数，即要划分多少次增量类

            incremental_labels = labels[original_class_num + l*incremental_class_num : (original_class_num + (l+1)*incremental_class_num)] #
            incremental_class_texts = []
            incremental_class_labels = []
            all_incremental_labels.append(incremental_labels)

            incremental_datanum_dict = {}  #初始化每个label对应的datanum数量
            for label in incremental_labels:
                incremental_datanum_dict[label] = 0 

            for line in range(datas_size):
                if datas['label'][line] in incremental_labels:

                    if incremental_datanum_dict[datas['label'][line]] >= data_num: #如果某个label的数量超过了设置的data_num，则跳过该label行
                        continue
                    incremental_datanum_dict[datas['label'][line]] += 1

                    incremental_class_texts.append(datas['text'][line])
                    incremental_class_labels.append(datas['label'][line])
            
            '''打乱训练数据集顺序'''
            if mode == 'train':
                randnum = random.randint(0,100)
                random.seed(randnum)
                random.shuffle(incremental_class_texts)
                random.seed(randnum)
                random.shuffle(incremental_class_labels)  #打乱训练集数据,这里注意必须打乱list类型的数据集，torch类型会导致重复
        
            incremental_dataframe = pd.DataFrame({'text':incremental_class_texts,'label':incremental_class_labels}) #将其储存为panda读取csv/tsv文件后的形式
            incremental_tensor_datas = self.My_DataLoader(incremental_dataframe, max_padding_size, label_to_idx)
            incremental_tensor_datas_list.append(incremental_tensor_datas)
        

        if drop_last == False:
            num = (len(labels) - original_class_num) / incremental_class_num
            num = int(num)
            if ((len(labels) - original_class_num) % incremental_class_num) != 0:  #如果不能整除
                last = original_class_num + num*incremental_class_num #整除情况的最后一个增量类从labels哪里开始
                incremental_last_labels = labels[last : len(labels)]
                all_incremental_labels.append(incremental_last_labels)

                incremental_datanum_dict = {}  #初始化每个label对应的datanum数量
                for label in incremental_last_labels:
                    incremental_datanum_dict[label] = 0 

                for line in range(datas_size):
                    if datas['label'][line] in incremental_last_labels:

                        if incremental_datanum_dict[datas['label'][line]] >= data_num: #如果某个label的数量超过了设置的data_num，则跳过该label行
                            continue
                        incremental_datanum_dict[datas['label'][line]] += 1

                        incremental_class_texts.append(datas['text'][line])
                        incremental_class_labels.append(datas['label'][line])
                '''打乱训练数据集顺序'''
                if mode == 'train':
                    randnum = random.randint(0,100)
                    random.seed(randnum)
                    random.shuffle(incremental_class_texts)
                    random.seed(randnum)
                    random.shuffle(incremental_class_labels)  #打乱训练集数据,这里注意必须打乱list类型的数据集，torch类型会导致重复
        
                incremental_dataframe = pd.DataFrame({'text':incremental_class_texts,'label':incremental_class_labels}) #将其储存为panda读取csv/tsv文件后的形式
                incremental_tensor_datas = self.My_DataLoader(incremental_dataframe, max_padding_size, label_to_idx)
                incremental_tensor_datas_list.append(incremental_tensor_datas)

        if mode == 'train': #训练模式下多返回一个labels，用于eval模式输入，保持测试集/验证集和训练集标签顺序一致
            return original_tensor_datas, incremental_tensor_datas_list, original_labels, all_incremental_labels, labels, label_to_idx
        elif mode == 'eval':
            return original_tensor_datas, incremental_tensor_datas_list, original_labels, all_incremental_labels

'''返回的数据original_tensor_datas是Dataloader封装的tensor类型数据，incremental_tensor_datas_list是一个列表，里面的元素是Datalodaer封装的tensor数据
original_labels为list, all_incremental_labels也为list，元素为增量类，labels为训练数据的所有标签，为list'''

'''训练集生成incremental类使用(drop_last=true丢弃最后不能整除的类数据)'''
#incremental = class_incremental('./data/snips/train.csv', 'csv', 128, 3, 2, True, 500)
#original_tensor_datas, incremental_tensor_datas_list, original_labels, all_incremental_labels, labels, label_to_idx = incremental.prepare_for_incremental()


'''测试集/验证集生成incremental类使用方式'''
#incremental = class_incremental('./data/snips/valid.csv', 'csv', 128, 3, 2,True, 500, 'eval', labels, label_to_idx)
#dev_tensor_datas, incremental_tensor_datas_list, dev_labels, all_incremental_labels = incremental.prepare_for_incremental()


'''观察每一个数据集的标签和对应的样本个数'''
def datanum_of_tags(datas_path, data_type):
    if data_type == 'csv':  
        datas = pd.read_csv(datas_path)
    elif data_type == 'tsv':
        datas = pd.read_csv(datas_path, sep='\t', header=0)
    las = []
    for i in range(datas.shape[0]):
        las.append(datas['label'][i]) 
    len(las)
    labels = []
    for i in range(datas.shape[0]):
        if las[i] not in labels:
            labels.append(las[i])

    for i in range(len(labels)):
        count = 0
        for j in range(len(las)):
            if las[j] == labels[i]:
                count+=1
        print(str(labels[i]) + ":" + str(count))

#使用方法： datanum_of_tags("./data/oos/test.tsv", 'tsv')