import os
import h5py
import pickle
from tqdm import tqdm

# 文件夹和 txt 文件路径
c3d_folder = '../data/c3d'
train_txt = '../data/vid_time3_train_new_2536_1233.txt'
val_txt = '../data/vid_time3_val_new_546_273.txt'
test_txt = '../data/vid_time3_test_new_542_304.txt'

# 读取 txt 文件，按顺序返回 vid 列表
def read_vids(txt_path):
    with open(txt_path, 'r') as f:
        vids = [line.strip() for line in f if line.strip()]
    return vids

train_vids = read_vids(train_txt)
val_vids = read_vids(val_txt)
test_vids = read_vids(test_txt)

# 初始化字典
def vids_to_dict(vid_list):
    vid_dict = {}
    for vid in tqdm(vid_list, desc=f"Processing {len(vid_list)} vids"):
        file_path = os.path.join(c3d_folder, f"{vid}.hdf5")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist!")
            continue
        try:
            with h5py.File(file_path, 'r') as f:
                group = f[vid]
                data = group['c3d_features'][:]
                vid_dict[vid] = data
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return vid_dict

# 分别生成字典
train_dict = vids_to_dict(train_vids)
val_dict = vids_to_dict(val_vids)
test_dict = vids_to_dict(test_vids)

# 保存成 pkl
with open('../data/c3d/c3d_train.pkl', 'wb') as f:
    pickle.dump(train_dict, f)
with open('../data/c3d/c3d_val.pkl', 'wb') as f:
    pickle.dump(val_dict, f)
with open('../data/c3d/c3d_test.pkl', 'wb') as f:
    pickle.dump(test_dict, f)

print("Done! Train/Val/Test dictionaries saved as pkl files.")
