import torch
import torch.nn as nn
import numpy as np
import sys
import random
import torch.backends.cudnn as cudnn
import torchmetrics

# 导入自定义模块
from Diffusion.Multimodal_Diffusion import GaussianDiffusionTrainer
from train import train, valid
from dataloader_fakesv import get_dataloader
from eval_metrics import eval_FakeSV

# 创建Tensorboard写入器
from tensorboardX import SummaryWriter
writer = SummaryWriter("logs")

def main(model_config = None):
    """
    主函数：训练和评估模型
    
    参数:
        model_config: 可选的模型配置 (默认为None)
    """
    # 默认模型配置
    modelConfig = {
        "state": "train",         # 训练状态
        "epoch": 60,              # 最大训练轮数
        "batch_size": 16,         # 批大小
        "T": 100,                 # 扩散时间步数
        "mult_dropout": 0.4,      # 多模态dropout
        "Text_Pre_dropout": 0.2,  # 文本预处理dropout
        "Img_Pre_dropout": 0.2,   # 图像预处理dropout
        "lr": 5e-6,               # 学习率
        "beta_1": 1e-4,           # 起始beta值
        "beta_T": 0.02,           # 终止beta值
        "device": "cuda:0",       # 训练设备
        # 输入维度配置
        "t_in": 768,              # 文本输入维度
        "i_in": 2048,             # 图像输入维度
        "a_in": 12288,            # 音频输入维度
        "v_in": 4096,             # 视觉输入维度
        "c3d_in": 4096,           # C3D特征维度
        # 预处理维度
        "t_in_pre": 100,          # 文本预处理后维度
        "a_in_pre": 128,           # 音频预处理后维度
        "v_in_pre": 128,           # 视觉预处理后维度
        "c3d_in_pre": 128,        # C3D预处理后维度
        "label_dim": 2,           # 标签维度 (二分类)
        "d_m": 128,               # 模态内部维度
        "unified_size": 128,      # 统一表示维度
        "vertex_num": 32,         # 胶囊图顶点数量
        "routing": 2,              # 路由迭代次数
        # 序列长度
        "T_t": 512,               # 文本序列长度
        "T_i": 49,                # 图像序列长度
        "T_a": 50,                # 音频序列长度
        "T_v": 83,                # 视觉序列长度
        "early_stop": 5,          # 早停轮数
        "weight_decay": 0.99,     # 权重衰减
        "datamode": 'title+ocr',  # 数据模式 (标题+OCR)
        "comments_dropout": 0.3,  # 评论特征dropout
        "dataset": 'SVFEND'       # 数据集名称
    }
    
    # 如果提供了配置，则更新默认配置
    if model_config is not None:
        modelConfig = model_config
        
    # 设置设备
    device = torch.device(modelConfig["device"])
    
    # 加载数据
    print("Start loading the data....")
    dataloader = get_dataloader(modelConfig=modelConfig, data_type='SVFEND')
    print('Finish loading the data....')
    
    # 创建模型
    trainer = GaussianDiffusionTrainer(
        modelConfig, 
        modelConfig["beta_1"], 
        modelConfig["beta_T"], 
        modelConfig["T"],
        modelConfig["t_in"], 
        modelConfig["a_in"], 
        modelConfig["v_in"], 
        modelConfig["d_m"], 
        modelConfig["mult_dropout"],
        modelConfig["label_dim"],
        modelConfig["unified_size"], 
        modelConfig["vertex_num"], 
        modelConfig["routing"], 
        modelConfig["T_t"],
        modelConfig["T_a"],  
        modelConfig["T_v"], 
        modelConfig["batch_size"]).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        trainer.parameters(), 
        lr=modelConfig["lr"], 
        weight_decay=modelConfig["weight_decay"])
    
    # 损失函数 (交叉熵损失)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 初始化最佳验证准确率和epoch计数器
    best_valid_acc = -1
    epoch, best_epoch = 0, 0
    
    # 训练循环
    while True:
        epoch += 1
        
        # ========== 训练阶段 ==========
        train_loss, train_results, train_truths = train(
            modelConfig, 
            dataloader["train"], 
            trainer, 
            criterion, 
            optimizer)
        
        # 计算训练指标
        train_acc, train_f1, train_pre, train_rec = eval_FakeSV(train_results, train_truths)
        
        # ========== 验证阶段 ==========
        valid_loss, valid_results, valid_truths = valid(
            dataloader["val"], 
            trainer, 
            criterion, 
            modelConfig)
        valid_acc, valid_f1, valid_pre, valid_rec = eval_FakeSV(valid_results, valid_truths)
        
        # ========== 测试阶段 ==========
        test_loss, test_results, test_truths = valid(
            dataloader["test"], 
            trainer, 
            criterion, 
            modelConfig)
        test_acc, test_f1, test_pre, test_rec = eval_FakeSV(test_results, test_truths)
        
        # ========== 记录指标 ==========
        # Tensorboard记录
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_f1', train_f1, epoch)
        writer.add_scalar('train_pre', train_pre, epoch)
        writer.add_scalar('train_rec', train_rec, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_f1', test_f1, epoch)
        writer.add_scalar('test_pre', test_pre, epoch)
        writer.add_scalar('test_rec', test_rec, epoch)
        writer.close()
        
        # 打印当前epoch结果
        print(f'Epoch {epoch:2d} Loss| Train Loss{train_loss:5.4f} | '
              f'Val Loss {valid_loss:5.4f} | Test Loss{test_loss:5.4f} || '
              f'Acc| Train Acc {train_acc:5.4f} | Val Acc {valid_acc:5.4f} | '
              f'Test Acc {test_acc:5.4f}')
        
        # 保存最佳模型 (当验证准确率提升时)
        if valid_acc >= (best_valid_acc + 1e-4):
            print(f'{modelConfig["dataset"]} dataset | acc improved! '
                  f'saving model to {modelConfig["dataset"]}_best_model.pkl')
            torch.save(trainer, f'{modelConfig["dataset"]}_best_model.pkl')
            
            # 更新最佳指标
            best_valid_acc = valid_acc
            best_acc = test_acc
            best_f1 = test_f1
            best_pre = test_pre
            best_rec = test_rec
            best_epoch = epoch
        
        # 早停检查 (当连续多个epoch未提升时停止)
        if epoch - best_epoch >= modelConfig["early_stop"]:
            break
        
        # 达到最大epoch数时停止
        if epoch > modelConfig["epoch"]:
            break
    
    # 打印最终结果
    print("Hyperparameter: ", modelConfig.items())
    print("Best Epoch:", best_epoch)
    print(f"Best Acc: {best_acc:5.4f}")
    print(f"F1: {best_f1:5.4f}")
    print(f"Precision: {best_pre:5.4f}")
    print(f"Recall: {best_rec:5.4f}")
    print('-' * 50)

def setup_seed(seed):
    """
    设置随机种子以保证可重复性
    
    参数:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

class Logger(object):
    """
    日志记录器类，将输出同时写入文件和标准输出
    
    参数:
        filename: 日志文件名
        stream: 标准输出流
    """
    def __init__(self, filename='default.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    # 设置随机种子
    setup_seed(2021)
    
    # 重定向输出到文件
    sys.stdout = Logger('result.txt', sys.stdout)
    sys.stderr = Logger('error.txt', sys.stderr)
    
    # 启动主函数
    main()