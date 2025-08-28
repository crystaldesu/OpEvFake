from typing import Dict
import torch
from tqdm import tqdm

def train(modelConfig: Dict, train_loader, trainer, criterion, optimizer):
    """
    训练一个epoch
    
    参数:
        modelConfig: 模型配置字典
        train_loader: 训练数据加载器
        trainer: 训练模型实例
        criterion: 损失函数 (交叉熵损失)
        optimizer: 优化器
        
    返回:
        total_loss: 本epoch的总损失
        results: 所有预测结果 (用于评估)
        truths: 所有真实标签 (用于评估)
    """
    results = []  # 存储每个批次的预测结果
    truths = []   # 存储每个批次的真实标签
    trainer.train()  # 设置模型为训练模式
    total_loss = 0.0  # 累计损失
    total_batch_size = 0  # 累计批次大小
    
    # 遍历训练数据 (显示进度条)
    for batch in tqdm(train_loader):
        batch_size = batch["label"].size(0)  # 当前批次大小
        # 从批次中提取各种模态数据
        texts = batch["text"]
        audios = batch["audioframes"]
        videos = batch["frames"]
        comments = batch["comments"]
        labels = batch["label"]
        c3d = batch["c3d"]
        user_intro = batch["user_intro"]
        gpt_description = batch["gpt_description"]
        
        total_batch_size += batch_size
        
        # 如果可用，将数据移动到GPU
        if torch.cuda.is_available():
            audios = audios.cuda()
            texts = texts.cuda()
            videos = videos.cuda()
            comments = comments.cuda()
            labels = labels.cuda()
            c3d = c3d.cuda()
            user_intro = user_intro.cuda()
            gpt_description = gpt_description.cuda()
            
        # 前向传播: 计算损失和预测
        loss, pred = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description)
        
        # 获取预测类别 (最大概率对应的类别)
        _, y = torch.max(pred, 1)
        
        # 计算总损失 (扩散损失 + 分类损失)
        diffusion_loss = loss.sum() / 1000.  # 扩散损失 (缩放)
        bce_loss_output = criterion(pred, labels)  # 分类损失 (交叉熵)
        
        # 组合损失 (带权重)
        loss_output = (torch.abs(bce_loss_output - 0.1) + 0.1) + diffusion_loss * 0.006
        
        # 记录结果和标签
        results.append(y)
        truths.append(labels)
        
        # 累计损失
        total_loss += loss_output.item()
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
    
    # 合并所有批次的预测结果和标签
    results = torch.cat(results)
    truths = torch.cat(truths)
    
    return total_loss, results, truths

def valid(loader, trainer, criterion, modelConfig: Dict):
    """
    验证或测试一个epoch
    
    参数:
        loader: 数据加载器 (验证或测试集)
        trainer: 模型实例
        criterion: 损失函数
        modelConfig: 模型配置字典
        
    返回:
        total_loss: 本epoch的总损失
        results: 所有预测结果
        truths: 所有真实标签
    """
    trainer.eval()  # 设置模型为评估模式
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    
    # 不计算梯度以节省内存
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_size = batch["label"].size(0)
            # 提取各种模态数据
            texts = batch["text"]
            audios = batch["audioframes"]
            videos = batch["frames"]
            comments = batch["comments"]
            labels = batch["label"]
            c3d = batch["c3d"]
            user_intro = batch["user_intro"]
            gpt_description = batch["gpt_description"]
            
            total_batch_size += batch_size
            
            # 如果可用，将数据移动到GPU
            if torch.cuda.is_available():
                audios = audios.cuda()
                texts = texts.cuda()
                videos = videos.cuda()
                comments = comments.cuda()
                labels = labels.cuda()
                c3d = c3d.cuda()
                user_intro = user_intro.cuda()
                gpt_description = gpt_description.cuda()
                
            # 前向传播
            loss, pred = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description)
            _, y = torch.max(pred, 1)
            
            # 计算损失 (同上)
            diffusion_loss = loss.sum() / 1000.
            bce_loss_output = criterion(pred, labels)
            loss_output = (torch.abs(bce_loss_output - 0.1) + 0.1) + diffusion_loss * 0.006
            
            # 记录结果
            results.append(y)
            truths.append(labels)
            total_loss += loss_output.item()
        
        # 合并所有结果
        results = torch.cat(results)
        truths = torch.cat(truths)
    
    return total_loss, results, truths