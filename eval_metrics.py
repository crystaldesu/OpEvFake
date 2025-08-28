import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

def metrics(y_true, y_pred):
    """
    计算多分类任务的评估指标
    
    参数:
        y_true: 真实标签数组
        y_pred: 预测概率数组或预测标签数组
        
    返回:
        metrics: 包含各项指标的字典
            - auc: 宏平均AUC
            - f1: 宏平均F1分数
            - recall: 宏平均召回率
            - precision: 宏平均精确率
            - acc: 准确率
    """
    metrics = {}
    # 计算宏平均AUC (需要预测概率)
    metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    
    # 将预测值四舍五入为整数标签
    y_pred = np.around(np.array(y_pred)).astype(int)
    
    # 计算宏平均F1分数
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    # 计算宏平均召回率
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    # 计算宏平均精确率
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    # 计算准确率
    metrics['acc'] = accuracy_score(y_true, y_pred)
    
    return metrics

def eval_FakeSV(results, truths):
    """
    专为FakeSV任务设计的评估函数
    
    参数:
        results: 预测结果张量 (torch.Tensor)
        truths: 真实标签张量 (torch.Tensor)
        
    返回:
        acc: 准确率
        f1: 宏平均F1分数
        precision: 宏平均精确率
        recall: 宏平均召回率
    """
    # 将张量移动到CPU并转换为numpy数组
    results = results.detach().cpu()
    truths = truths.detach().cpu()
    
    # 调用metrics函数计算指标
    results_metrics = metrics(truths, results)
    
    # 返回关键指标
    return (results_metrics['acc'], 
            results_metrics['f1'], 
            results_metrics['precision'], 
            results_metrics['recall'])