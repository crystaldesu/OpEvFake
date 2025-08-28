This is an executable version of https://github.com/ZhouJiahui-dlut/OpEvFake

The dataset is provided, and Chinese annotations have been added to the code.

## Environment:

Python 3.9

PyTorch 2.2.2

CUDA 11.8

## Dataset Download:



## Run the Code

1. Download the dataset above and extract them to the 'data/' directory. 
2. Command as follows.

```
python main.py
```

## Notes:

1. The original paper used ChatGPT 3.5 as the LLM, while the dataset provided in this repository was generated using DeepSeek-V3.1 (Non-thinking Mode). Therefore, the actual trained model may exhibit differences in metrics such as F1 score/Accuracy compared to the results reported in the original paper. Please note this discrepancy.

