#!/usr/bin/env python3
"""
BERT + AG News完整数据集实验：标准BERT vs ELM风格BERT（冻结Q,K）
AG News数据集：120,000训练样本 + 7,600测试样本，4分类任务
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import json
import argparse
import os
import csv


class AGNewsDataset(Dataset):
    """AG News Dataset for BERT"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_agnews_data(split='train', max_samples=None):
    """加载AG News数据集"""
    data_dir = 'data/ag_news_csv'
    filepath = os.path.join(data_dir, f'{split}.csv')
    
    if not os.path.exists(filepath):
        print(f"错误：数据文件不存在: {filepath}", flush=True)
        raise FileNotFoundError(f"请确保 {filepath} 存在")
    
    texts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                label = int(row[0]) - 1  # AG News标签是1-4，转换为0-3
                title = row[1]
                description = row[2]
                text = title + ' ' + description
                texts.append(text)
                labels.append(label)
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    print(f"加载 {split} 数据: {len(texts)} 样本", flush=True)
    return texts, labels


def freeze_bert_qk_projections(model):
    """冻结BERT所有层的Q和K投影矩阵"""
    frozen_params = 0
    for name, param in model.named_parameters():
        if 'attention' in name and ('query' in name or 'key' in name):
            param.requires_grad = False
            frozen_params += param.numel()
    return frozen_params


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def run_single_experiment(train_texts, train_labels, test_texts, test_labels, 
                         tokenizer, epochs, batch_size, learning_rate, 
                         device, freeze_qk=False, seed=42):
    """运行单次实验"""
    torch.manual_seed(seed)
    
    # 创建数据集和数据加载器
    train_dataset = AGNewsDataset(train_texts, train_labels, tokenizer)
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model.to(device)
    
    # 冻结Q,K投影（如果需要）
    frozen_params = 0
    if freeze_qk:
        frozen_params = freeze_bert_qk_projections(model)
    
    total_params, trainable_params = count_parameters(model)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        
        print(f"      Epoch {epoch+1}/{epochs}... Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%", flush=True)
    
    training_time = time.time() - start_time
    
    return {
        'accuracy': best_acc * 100,
        'time': training_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params
    }


def main():
    parser = argparse.ArgumentParser(description='BERT AG News实验')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--num-exp', type=int, default=10, help='重复实验次数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数（用于快速测试）')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("BERT实验：标准BERT vs ELM风格BERT（冻结Q,K）", flush=True)
    print(f"数据集: AG News (完整数据集)", flush=True)
    print(f"实验次数: {args.num_exp}", flush=True)
    print(f"基础随机种子: 42", flush=True)
    print(f"训练轮数: {args.epochs}", flush=True)
    print(f"批次大小: {args.batch_size}", flush=True)
    print("="*70 + "\n", flush=True)
    
    # 加载数据
    if args.max_samples:
        print(f"使用采样数据: {args.max_samples} 训练样本", flush=True)
    else:
        print("加载AG News完整数据集...", flush=True)
    
    train_texts, train_labels = load_agnews_data('train', max_samples=args.max_samples)
    test_texts, test_labels = load_agnews_data('test')
    
    print(f"训练样本: {len(train_texts)}", flush=True)
    print(f"测试样本: {len(test_texts)}", flush=True)
    
    # 加载tokenizer
    print("\n加载BERT tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 存储结果
    results = {
        'standard': {
            'accuracies': [],
            'times': [],
            'params': 0,
            'trainable_params': 0
        },
        'elm': {
            'accuracies': [],
            'times': [],
            'params': 0,
            'trainable_params': 0,
            'frozen_params': 0
        }
    }
    
    # 运行实验
    base_seed = 42
    for exp_num in range(args.num_exp):
        seed = base_seed + exp_num * 100
        
        print("\n" + "="*70, flush=True)
        print(f"实验 {exp_num+1}/{args.num_exp} (seed={seed})", flush=True)
        print("="*70 + "\n", flush=True)
        
        # 标准BERT
        print("测试 标准BERT...", flush=True)
        standard_result = run_single_experiment(
            train_texts, train_labels, test_texts, test_labels,
            tokenizer, args.epochs, args.batch_size, args.lr,
            device, freeze_qk=False, seed=seed
        )
        results['standard']['accuracies'].append(standard_result['accuracy'])
        results['standard']['times'].append(standard_result['time'])
        results['standard']['params'] = standard_result['total_params']
        results['standard']['trainable_params'] = standard_result['trainable_params']
        
        print(f"最佳准确率: {standard_result['accuracy']:.2f}%, "
              f"训练时间: {standard_result['time']:.2f}秒, "
              f"可训练参数: {standard_result['trainable_params']:,}", flush=True)
        
        # ELM-BERT
        print("\n测试 ELM风格BERT (冻结Q,K)...", flush=True)
        elm_result = run_single_experiment(
            train_texts, train_labels, test_texts, test_labels,
            tokenizer, args.epochs, args.batch_size, args.lr,
            device, freeze_qk=True, seed=seed
        )
        results['elm']['accuracies'].append(elm_result['accuracy'])
        results['elm']['times'].append(elm_result['time'])
        results['elm']['params'] = elm_result['total_params']
        results['elm']['trainable_params'] = elm_result['trainable_params']
        results['elm']['frozen_params'] = elm_result['frozen_params']
        
        print(f"最佳准确率: {elm_result['accuracy']:.2f}%, "
              f"训练时间: {elm_result['time']:.2f}秒", flush=True)
        print(f"可训练参数: {elm_result['trainable_params']:,} "
              f"(冻结: {elm_result['frozen_params']:,})", flush=True)
        
        # 输出本次对比
        param_reduction = (elm_result['frozen_params'] / standard_result['total_params']) * 100
        time_diff = elm_result['time'] - standard_result['time']
        time_diff_pct = (time_diff / standard_result['time']) * 100
        acc_diff = elm_result['accuracy'] - standard_result['accuracy']
        
        print(f"\n本次实验对比:", flush=True)
        print(f"  参数节省: {param_reduction:.2f}%", flush=True)
        print(f"  准确率差异: {acc_diff:+.2f}%", flush=True)
        print(f"  训练时间差异: {time_diff:+.2f}秒 ({time_diff_pct:+.2f}%)", flush=True)
    
    # 计算统计结果
    import numpy as np
    
    print("\n" + "="*70, flush=True)
    print("最终统计结果", flush=True)
    print("="*70 + "\n", flush=True)
    
    # 标准BERT统计
    std_acc_mean = np.mean(results['standard']['accuracies'])
    std_acc_std = np.std(results['standard']['accuracies'])
    std_time_mean = np.mean(results['standard']['times'])
    std_time_std = np.std(results['standard']['times'])
    
    print(f"标准BERT:", flush=True)
    print(f"  准确率: {std_acc_mean:.2f}% ± {std_acc_std:.2f}%", flush=True)
    print(f"  训练时间: {std_time_mean:.2f}s ± {std_time_std:.2f}s", flush=True)
    print(f"  总参数: {results['standard']['params']:,}", flush=True)
    print(f"  可训练参数: {results['standard']['trainable_params']:,}", flush=True)
    
    # ELM-BERT统计
    elm_acc_mean = np.mean(results['elm']['accuracies'])
    elm_acc_std = np.std(results['elm']['accuracies'])
    elm_time_mean = np.mean(results['elm']['times'])
    elm_time_std = np.std(results['elm']['times'])
    
    print(f"\nELM-BERT:", flush=True)
    print(f"  准确率: {elm_acc_mean:.2f}% ± {elm_acc_std:.2f}%", flush=True)
    print(f"  训练时间: {elm_time_mean:.2f}s ± {elm_time_std:.2f}s", flush=True)
    print(f"  总参数: {results['elm']['params']:,}", flush=True)
    print(f"  可训练参数: {results['elm']['trainable_params']:,}", flush=True)
    print(f"  冻结参数: {results['elm']['frozen_params']:,}", flush=True)
    
    # 对比统计
    param_reduction = (results['elm']['frozen_params'] / results['standard']['params']) * 100
    acc_diff = elm_acc_mean - std_acc_mean
    time_diff = elm_time_mean - std_time_mean
    time_diff_pct = (time_diff / std_time_mean) * 100
    
    print(f"\n总体对比:", flush=True)
    print(f"  参数节省: {param_reduction:.2f}%", flush=True)
    print(f"  准确率差异: {acc_diff:+.2f}%", flush=True)
    print(f"  训练时间差异: {time_diff:+.2f}秒 ({time_diff_pct:+.2f}%)", flush=True)
    
    # 保存结果
    output_file = 'bert_agnews_experiments_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n结果已保存到: {output_file}", flush=True)


if __name__ == '__main__':
    main()
