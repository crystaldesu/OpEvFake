# audio_data_extract.py
import pickle

# 训练/验证/测试划分txt
split_files = {
    "train": "../data/temporal_new/vid_time3_train_new_2536_1233.txt",
    "val": "../data/temporal_new/vid_time3_val_new_546_273.txt",
    "test": "../data/temporal_new/vid_time3_test_new_542_304.txt",
}

# 原始音频特征
with open("../data/dict_vid_audioconvfea.pkl", "rb") as f:
    audio_dict = pickle.load(f)

# 逐份划分并保存
for split, txt_path in split_files.items():
    with open(txt_path, "r", encoding="utf-8") as f:
        vids = [line.strip() for line in f.readlines()]
    split_dict = {vid: audio_dict[vid] for vid in vids if vid in audio_dict}

    with open(f"../data/audio/audio_{split}.pkl", "wb") as f:
        pickle.dump(split_dict, f)
    print(f"audio_{split}.pkl 保存完成, 共 {len(split_dict)} 条")
