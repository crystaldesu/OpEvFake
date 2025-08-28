# label_extract.py
import pickle

def extract_labels(data_pkl, output_file):
    with open(data_pkl, "rb") as f:
        dataset = pickle.load(f)

    label_map = {"真": 1, "假": 0}
    labels = {}

    for item in dataset:
        vid = item["video_id"]
        ann = item["annotation"]
        if ann not in label_map:
            print(f"Warning: unknown label {ann} for {vid}, skip")
            continue
        labels[vid] = label_map[ann]

    with open(output_file, "wb") as f:
        pickle.dump(labels, f)
    print(f"Saved labels to {output_file}")


if __name__ == "__main__":
    extract_labels("../data/data_train_list.pkl", "../data/label/label_train.pkl")
    extract_labels("../data/data_val_list.pkl", "../data/label/label_val.pkl")
    extract_labels("../data/data_test_list.pkl", "../data/label/label_test.pkl")
