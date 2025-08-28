import math
import os
import pickle

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertTokenizer
from torchvision import models
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

# 定义一个视频对应的所有数据
class SVFENDDataset(Dataset):
    """SVFEND 数据集加载器"""
    
    def __init__(self, datamode='title+ocr', train_or_test='train'):
        """
        初始化数据集
        
        参数:
            datamode: 数据模式 ('title' 或 'title+ocr')
            train_or_test: 数据集类型 ('train', 'val', 'test')
        """
        # 读取音频特征
        with open(os.path.join('./data/audio', 'audio_' + train_or_test + '.pkl'), "rb") as fr:
            audio = pickle.load(fr)

        # 读取文本特征 (根据模式选择)
        if datamode == 'title':
            with open(os.path.join('./data/text_title_temporal', 'text_title_lhs_' + train_or_test + '.pkl'), "rb") as fr:
                self.text = pickle.load(fr)
        elif datamode == 'title+ocr':
            with open(os.path.join('./data/text_title_ocr_temporal', 'text_title_ocr_lhs_' + train_or_test + '.pkl'), "rb") as fr:
                self.text = pickle.load(fr)

        # 读取其他特征
        with open(os.path.join('./data/comments', 'comments_' + train_or_test + '.pkl'), "rb") as fr:
            self.comments = pickle.load(fr)
        with open(os.path.join('./data/gpt', 'gpt_description_' + train_or_test + '.pkl'), "rb") as fr:
            self.gpt_description = pickle.load(fr)
        with open(os.path.join('./data/label', 'label_' + train_or_test + '.pkl'), "rb") as fr:
            self.label = pickle.load(fr)
        with open(os.path.join('./data/video', 'video_' + train_or_test + '.pkl'), "rb") as fr:
            self.video = pickle.load(fr)
        with open(os.path.join('./data/user_intro', 'user_intro_' + train_or_test + '.pkl'), "rb") as fr:
            self.user_intro = pickle.load(fr)
        with open(os.path.join('./data/vid', 'vid_' + train_or_test + '.pkl'), "rb") as fr:
            self.vid = pickle.load(fr)
        with open(os.path.join('./data/c3d', 'c3d_' + train_or_test + '.pkl'), "rb") as fr:
            self.c3d = pickle.load(fr)

        # 过滤音频特征 (只保留vid中存在的)
        self.audio = dict(filter(lambda item: item[0] in self.vid, audio.items()))

    def __len__(self):
        """返回数据集大小"""
        return len(self.label)

    def __getitem__(self, idx):
        """获取单个样本"""
        vid = self.vid[idx]  # 视频ID
        
        # 获取各种特征
        text = torch.tensor(self.text[vid], dtype=torch.float32)
        comments = self.comments[vid]
        gpt_description = self.gpt_description[vid]
        audio = self.audio[vid]
        video = self.video[vid]
        c3d = self.c3d[vid]
        label = self.label[vid]
        label = torch.tensor(label)
        user_intro = self.user_intro[vid]
        
        # 转换为张量
        audio = torch.tensor(audio, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)
        c3d = torch.tensor(c3d, dtype=torch.float32)

        return {
            'label': label,
            'text': text,
            'audioframes': audio,
            'frames': video,
            'comments': comments,
            'c3d': c3d,
            'user_intro': user_intro,
            'gpt_description': gpt_description
        }

def pad_sequence(seq_len, lst, emb):
    """填充序列到固定长度"""
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        # 处理空序列
        if ori_len == 0:
            video = torch.zeros([seq_len, emb], dtype=torch.long)
        # 处理长序列
        elif ori_len >= seq_len:
            if emb == 200:
                video = torch.FloatTensor(video[:seq_len])
            else:
                video = torch.LongTensor(video[:seq_len])
        # 处理短序列
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
            if emb == 200:
                video = torch.FloatTensor(video)
            else:
                video = torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len, lst):
    """填充帧序列到固定长度并创建注意力掩码"""
    attention_masks = []
    result = []
    for video in lst:
        video = video.squeeze()
        ori_len = video.shape[0]
        # 处理长序列
        if ori_len >= seq_len:
            gap = ori_len // seq_len  # 计算采样间隔
            video = video[::gap][:seq_len]  # 均匀采样
            mask = np.ones((seq_len))  # 全1掩码
        # 处理短序列
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float32)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))  # 部分1掩码
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def SVFEND_collate_fn(batch):
    """SVFEND数据集的批处理函数"""
    num_frames = 83  # 视觉帧数
    num_audioframes = 50  # 音频帧数

    # 提取并处理各种特征
    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)
    frames = frames.squeeze()

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    comments = [item['comments'] for item in batch]
    comments = torch.stack(comments)

    gpt_description = [item['gpt_description'] for item in batch]
    gpt_description = torch.stack(gpt_description)

    user_intro = [item['user_intro'] for item in batch]
    user_intro = torch.stack(user_intro)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]
    text = [item['text'] for item in batch]
    text = torch.stack(text)

    return {
        'label': torch.stack(label),
        'text': text,
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'comments': comments,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
        'user_intro': user_intro,
        'gpt_description': gpt_description,
    }

def _init_fn(worker_id):
    """数据加载器工作进程初始化函数 (设置随机种子)"""
    np.random.seed(2022)

def get_dataloader(modelConfig, data_type='SVFEND'):
    """获取数据加载器"""
    collate_fn = None

    if data_type == 'SVFEND':
        # 创建数据集
        dataset_train = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='train')
        dataset_val = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='val')
        dataset_test = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='test')
        collate_fn = SVFEND_collate_fn  # 设置批处理函数

    # 创建数据加载器
    train_dataloader = DataLoader(dataset_train, batch_size=modelConfig["batch_size"],
                                  num_workers=0, pin_memory=True, shuffle=True,
                                  worker_init_fn=_init_fn, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset_val, batch_size=modelConfig["batch_size"],
                                num_workers=0, pin_memory=True, shuffle=False,
                                worker_init_fn=_init_fn, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset_test, batch_size=modelConfig["batch_size"],
                                 num_workers=0, pin_memory=True, shuffle=False,
                                 worker_init_fn=_init_fn, collate_fn=collate_fn)

    # 返回数据加载器字典
    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

def split_word(df):  # 去除停用词
    title = df['description'].values
    comments = df['comments'].apply(lambda x: ' '.join(x)).values
    text = np.concatenate([title, comments], axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus