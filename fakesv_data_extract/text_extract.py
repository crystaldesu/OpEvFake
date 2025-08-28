# text_extract.py
import pickle
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

PTM = "./bert-base-chinese/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, config, datamode="title+ocr"):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_size = config.pad_size
        self.max_sentence_len = 0
        self.datamode = datamode

        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item["video_id"]

        # 根据 datamode 拼接文本
        if self.datamode == "title":
            text = item["title"]
        else:
            text = item["title"] + " " + item.get("ocr", "")

        input_ids, attention_mask = self.__convert_to_id__(text)

        return torch.tensor(input_ids), torch.tensor(attention_mask), vid

    def __convert_to_id__(self, sentence):
        ids = self.tokenizer.encode_plus(sentence, max_length=self.pad_size, truncation=True)
        input_ids = self.__padding__(ids['input_ids'])
        attention_mask = self.__padding__(ids['attention_mask'])
        return input_ids, attention_mask

    def __padding__(self, sentence):
        if self.max_sentence_len < len(sentence):
            self.max_sentence_len = len(sentence)
        sentence = sentence[:self.pad_size]
        sentence = sentence + [0] * (self.pad_size - len(sentence))
        return sentence


class TextExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)
        # 添加线性层将768维BERT输出转换为100维
        self.fc = nn.Linear(768, 100)

    def forward(self, x):
        # 获取所有token的表示
        x_bert = self.bert(input_ids=x[0], attention_mask=x[1]).last_hidden_state
        # 应用线性层将768维转换为100维
        x_bert = self.fc(x_bert)
        return x_bert, x[2]


def extract_text_features(data_pkl, output_file, datamode="title+ocr"):
    with open(data_pkl, "rb") as f:
        dataset = pickle.load(f)

    class Config:
        def __init__(self):
            self.PTM = PTM
            self.pad_size = 512  # 将序列长度从100改为512，与GPT特征匹配
            self.batch_size = 8

    config = Config()

    tokenizer = BertTokenizer.from_pretrained(PTM)
    text_dataset = TextDataset(dataset, tokenizer, config, datamode)
    dataloader = DataLoader(text_dataset, batch_size=config.batch_size)

    model = TextExtractor(config).to(device)
    model.eval()

    text_features = {}

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting text features"):
            # 将数据移动到设备
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            vids = data[2]

            # 获取所有token的表示
            outputs, _ = model((input_ids, attention_mask, vids))

            # 保存每个视频的特征
            for i, vid in enumerate(vids):
                text_features[vid] = outputs[i].cpu()  # 形状为 [seq_len, 100]

    with open(output_file, "wb") as f:
        pickle.dump(text_features, f)
    print(f"Saved text features to {output_file}")


if __name__ == "__main__":
    extract_text_features("../data/data_train_list.pkl", "../data/text_title_ocr_temporal/text_title_ocr_lhs_train.pkl")
    extract_text_features("../data/data_val_list.pkl", "../data/text_title_ocr_temporal/text_title_ocr_lhs_val.pkl")
    extract_text_features("../data/data_test_list.pkl", "../data/text_title_ocr_temporal/text_title_ocr_lhs_test.pkl")