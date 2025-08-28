# BERT
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time
import pickle


def read_json(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            data.append(json.loads(line))
    return data


def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1]) * 10000
    elif '亿' in str_x:
        return float(str_x[:-1]) * 100000000
    else:
        print("error")
        print(str_x)
        return 0  # 转换失败时返回0


def pad_sequence(seq_len, video, emb):
    if isinstance(video, list):
        video = torch.stack(video)
    ori_len = video.shape[0]
    if ori_len == 0:
        video = torch.zeros([seq_len, emb], dtype=torch.long)
    elif ori_len >= seq_len:
        if emb == 200:
            video = torch.FloatTensor(video[:seq_len])
        else:
            video = torch.LongTensor(video[:seq_len])
    else:
        video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
        if emb == 200:
            video = torch.FloatTensor(video)
        else:
            video = torch.LongTensor(video)
    return video


class BaselineData(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_size = config.pad_size  # 510
        self.max_sentence_len = 0
        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comments_like = []
        for num in self.data[idx]['count_comment_like']:
            num_like = num.split(" ")[0]
            comments_like.append(str2num(num_like))

        comments_inputid = []
        comments_mask = []

        # 检查评论列表是否为空
        if len(self.data[idx]['comments']) == 0:
            # 评论列表为空，使用全零填充
            comments_inputid = torch.zeros((23, self.pad_size), dtype=torch.long)
            comments_mask = torch.zeros((23, self.pad_size), dtype=torch.long)
            comments_like = torch.zeros(23, dtype=torch.long)
        else:
            for comment in self.data[idx]['comments']:
                comment_tokens = self.__convert_to_id__(comment)
                comments_inputid.append(comment_tokens[0])
                comments_mask.append(comment_tokens[1])

            num_comments = 23
            comments_inputid = torch.LongTensor(np.array(comments_inputid))
            comments_mask = torch.LongTensor(np.array(comments_mask))

            comments_inputid = pad_sequence(num_comments, comments_inputid, self.pad_size)
            comments_mask = pad_sequence(num_comments, comments_mask, self.pad_size)

            if len(comments_like) >= num_comments:
                comments_like = torch.tensor(comments_like[:num_comments])
            else:
                comments_like = torch.tensor(comments_like + [0] * (num_comments - len(comments_like)))

        return {
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'video_id': self.data[idx]['video_id']
        }

    def __convert_to_id__(self, sentence):
        ids = self.tokenizer.encode_plus(sentence, max_length=self.pad_size, truncation=True)
        input_ids = self.__padding__(ids['input_ids'])
        attention_mask = self.__padding__(ids['attention_mask'])
        return input_ids, attention_mask

    def __padding__(self, sentence):
        if self.max_sentence_len < len(sentence):
            self.max_sentence_len = len(sentence)
            print(self.max_sentence_len)
        sentence = sentence[:self.pad_size]
        sentence = sentence + [0] * (self.pad_size - len(sentence))
        return sentence


class Config():
    def __init__(self):
        self.pad_size = 250
        self.batch_size = 8
        self.epochs = 1
        self.PTM = './bert-base-chinese/'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()

vid_list = []
with open('../data/vid_time3_train.txt', 'r', encoding='utf-8') as f:
    for vid in f.readlines():
        vid = vid.strip('\n')
        vid_list.append(vid)

with open('../data/data_train_list.pkl', 'rb') as f:
    fakesv_data_new = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained(config.PTM)
dataloader = DataLoader(BaselineData(fakesv_data_new, tokenizer, config), batch_size=config.batch_size)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)

    def forward(self, x):
        # 获取设备信息
        device = next(self.parameters()).device

        # 确保comments_feature在正确的设备上
        comments_feature = torch.empty(size=(0, 23, 768), dtype=torch.float32).to(device)

        for i in range(x['comments_inputid'].shape[0]):
            # 将输入数据移动到GPU
            input_ids = x['comments_inputid'][i].to(device)
            attention_mask = x['comments_mask'][i].to(device)

            # 检查是否所有输入都是填充的（没有真实评论）
            if torch.all(input_ids == 0):
                # 没有真实评论，使用零向量
                bert_fea = torch.zeros(23, 768, device=device).unsqueeze(0)
            else:
                # 有真实评论，使用BERT提取特征
                bert_fea = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
                bert_fea = bert_fea.unsqueeze(0)

            comments_feature = torch.cat([comments_feature, bert_fea])

        # 确保fea_comments在正确的设备上
        fea_comments = torch.empty(size=(0, 768), dtype=torch.float32).to(device)

        for v in range(x['comments_like'].shape[0]):
            # 将comments_like移动到GPU
            comments_like_v = x['comments_like'][v].to(device)

            # 检查是否有评论
            if torch.all(comments_like_v == 0):
                # 没有评论或点赞数为零，使用均匀权重
                comments_weight = torch.ones(23, device=device) / 23
            else:
                # 计算权重，避免除零错误
                denominator = comments_like_v.shape[0] + comments_like_v.sum()
                if denominator == 0:
                    comments_weight = torch.ones(23, device=device) / 23
                else:
                    comments_weight = torch.stack(
                        [torch.true_divide((i + 1), denominator) for i in comments_like_v])

            # 检查comments_weight是否有NaN
            if torch.isnan(comments_weight).any():
                # 如果有NaN，使用均匀权重
                comments_weight = torch.ones(23, device=device) / 23

            comments_fea_reweight = torch.sum(
                comments_feature[v] * (comments_weight.reshape(comments_weight.shape[0], 1)), dim=0)
            comments_fea_reweight = comments_fea_reweight.unsqueeze(0)
            fea_comments = torch.cat([fea_comments, comments_fea_reweight])

        return fea_comments, x['video_id']


Model = Model(config)
Model.to(config.device)
text_data = {}
Model.eval()

with torch.no_grad():
    for data in tqdm(dataloader, desc="Extracting BERT features"):
        # 将数据移动到GPU
        data_on_device = {}
        for key, value in data.items():
            if key != 'video_id':
                data_on_device[key] = value.to(config.device)
            else:
                data_on_device[key] = value

        y, vid = Model(data_on_device)

        # 检查是否有NaN，如果有则替换为零向量
        if torch.isnan(y).any():
            print(f"WARNING: NaN detected in features for videos {vid}")
            y = torch.zeros_like(y)

        # 将结果移回CPU以便保存
        y = y.cpu()

        for idx, data_vid in enumerate(vid):
            text_data[data_vid] = y[idx]

with open('../data/comments/comments_train.pkl', 'wb') as f:
    pickle.dump(text_data, f)
    print("保存成功")