# video_data_extract.py
import os
import pickle

split_files = {
    "train": "./data/temporal_new/vid_time3_train_new_2536_1233.txt",
    "val": "./data/temporal_new/vid_time3_val_new_546_273.txt",
    "test": "./data/temporal_new/vid_time3_test_new_542_304.txt",
}

video_dir = "./data/ptvgg19_frames/"

def load_video_from_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

for split, txt_path in split_files.items():
    with open(txt_path, "r", encoding="utf-8") as f:
        vids = [line.strip() for line in f.readlines()]
    split_dict = {}
    for vid in vids:
        pkl_path = os.path.join(video_dir, f"{vid}.pkl")
        if os.path.exists(pkl_path):
            split_dict[vid] = load_video_from_pkl(pkl_path)
    with open(f"./data/video/video_{split}.pkl", "wb") as f:
        pickle.dump(split_dict, f)
    print(f"video_{split}.pkl 保存完成, 共 {len(split_dict)} 条")
