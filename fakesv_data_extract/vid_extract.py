# vid_data_extract.py
import pickle

# 三个划分 txt
split_files = {
    "train": "../data/temporal_new/vid_time3_train_new_2536_1233.txt",
    "val": "../data/temporal_new/vid_time3_val_new_546_273.txt",
    "test": "../data/temporal_new/vid_time3_test_new_542_304.txt",
}

for split, txt_path in split_files.items():
    with open(txt_path, "r", encoding="utf-8") as f:
        vids = [line.strip() for line in f.readlines()]

    with open(f"../data/vid/vid_{split}.pkl", "wb") as f:
        pickle.dump(vids, f)

    print(f"vid_{split}.pkl 保存完成，共 {len(vids)} 条")
