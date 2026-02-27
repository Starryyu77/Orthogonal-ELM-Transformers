#!/usr/bin/env python3
"""
BERT + IMDB完整数据集实验：标准BERT vs ELM风格BERT (冻结Q,K)
使用完整25000训练样本 + 25000测试样本
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
import json
import argparse
import os
from scipy import stats
import numpy as np


class IMDBDataset(Dataset):
    """IMDB Dataset for BERT"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_data(data_dir='data/aclImdb', split='train'):
    """加载IMDB数据集"""
    texts = []
    labels = []
    
    split_dir = os.path.join(data_dir, split)
    
    # 加载正面评论
    pos_dir = os.path.join(split_dir, 'pos')
    for filename in os.listdir(pos_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)
    
    # 加载负面评论
    neg_dir = os.path.join(split_dir, 'neg')
    for filename in os.listdir(neg_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)
    
    return texts, labels


def freeze_bert_qk_projections(model):
    """冻结BERT所有attention层的Q,K投影"""
    frozen_params = 0
    for layer in model.bert.encoder.layer:
        # 冻结query投影
        layer.attention.self.query.weight.requires_grad = False
        layer.attention.self.query.bias.requires_grad = False
        frozen_params += layer.attention.self.query.weight.numel()
        frozen_params += layer.attention.self.query.bias.numel()
        
        # 冻结key投影
        layer.attention.self.key.weight.requires_grad = False
        layer.attention.self.key.bias.requires_grad = False
        frozen_params += layer.attention.self.key.weight.numel()
        frozen_params += layer.attention.self.key.bias.numel()
    
    return frozen_params


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def run_single_experiment(train_loader, test_loader, device, seed, freeze_qk=False, num_epochs=3, lr=2e-5):
    """运行单次实验"""
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 加载BERT模型
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    ).to(device)
    
    # 如果需要，冻结Q,K投影
    frozen_params = 0
    if freeze_qk:
        frozen_params = freeze_bert_qk_projections(model)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练
    best_acc = 0
    training_time = 0
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        training_time += time.time() - epoch_start
        
        best_acc = max(best_acc, test_acc)
        
        print(f"    Epoch {epoch}/{num_epochs}...", flush=True)
        print(f"      Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%", flush=True)
    
    return best_acc, training_time, total_params, trainable_params, frozen_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='基础随机种子')
    parser.add_argument('--num-exp', type=int, default=10, help='实验次数')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max-length', type=int, default=256, help='最大序列长度')
    parser.add_argument('--max-samples', type=int, default=None, help='最大训练样本数（用于快速测试）')
    args = parser.parse_args()
    
    base_seed = args.seed
    num_experiments = args.num_exp
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}", flush=True)
    
    print("", flush=True)
    print("="*70, flush=True)
    print("BERT实验：标准BERT vs ELM风格BERT（冻结Q,K）", flush=True)
    print(f"数据集: IMDB (完整数据集)", flush=True)
    print(f"实验次数: {num_experiments}", flush=True)
    print(f"基础随机种子: {base_seed}", flush=True)
    print(f"训练轮数: {args.epochs}", flush=True)
    print(f"批次大小: {args.batch_size}", flush=True)
    print("="*70, flush=True)
    print("", flush=True)
    
    # 加载IMDB数据集
    print("加载IMDB完整数据集...", flush=True)
    train_texts, train_labels = load_imdb_data(split='train')
    test_texts, test_labels = load_imdb_data(split='test')
    
    # 如果指定了最大样本数，进行采样
    if args.max_samples is not None and args.max_samples < len(train_texts):
        print(f"使用采样数据: {args.max_samples} 训练样本", flush=True)
        indices = list(range(len(train_texts)))
        np.random.seed(base_seed)
        np.random.shuffle(indices)
        indices = indices[:args.max_samples]
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
        # 测试集也采样
        test_indices = list(range(len(test_texts)))
        np.random.shuffle(test_indices)
        test_indices = test_indices[:args.max_samples // 5]
        test_texts = [test_texts[i] for i in test_indices]
        test_labels = [test_labels[i] for i in test_indices]
    
    print(f"训练样本: {len(train_texts)}", flush=True)
    print(f"测试样本: {len(test_texts)}", flush=True)
    print("", flush=True)
    
    # 加载tokenizer
    print("加载BERT tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("", flush=True)
    
    # 创建数据集和数据加载器（只创建一次，重复使用）
    print("创建数据加载器...", flush=True)
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练批次数: {len(train_loader)}", flush=True)
    print(f"测试批次数: {len(test_loader)}", flush=True)
    print("", flush=True)
    
    # 记录结果
    results = {
        'standard': {'accuracies': [], 'times': [], 'params': None, 'trainable_params': None},
        'elm': {'accuracies': [], 'times': [], 'params': None, 'trainable_params': None, 'frozen_params': None}
    }
    
    # 运行重复实验
    for exp_idx in range(num_experiments):
        seed = base_seed + exp_idx * 100
        
        print("="*70, flush=True)
        print(f"实验 {exp_idx + 1}/{num_experiments} (seed={seed})", flush=True)
        print("="*70, flush=True)
        print("", flush=True)
        
        # 测试标准BERT
        print("测试 标准BERT...", flush=True)
        std_acc, std_time, std_total, std_trainable, _ = run_single_experiment(
            train_loader, test_loader, device, seed,
            freeze_qk=False, num_epochs=args.epochs, lr=args.lr
        )
        results['standard']['accuracies'].append(std_acc * 100)
        results['standard']['times'].append(std_time)
        if results['standard']['params'] is None:
            results['standard']['params'] = std_total
            results['standard']['trainable_params'] = std_trainable
        
        print(f"  最佳准确率: {std_acc*100:.2f}%", flush=True)
        print(f"  训练时间: {std_time:.2f}秒", flush=True)
        print(f"  可训练参数: {std_trainable:,}", flush=True)
        print("", flush=True)
        
        # 测试ELM风格BERT（冻结Q,K）
        print("测试 ELM风格BERT (冻结Q,K)...", flush=True)
        elm_acc, elm_time, elm_total, elm_trainable, elm_frozen = run_single_experiment(
            train_loader, test_loader, device, seed,
            freeze_qk=True, num_epochs=args.epochs, lr=args.lr
        )
        results['elm']['accuracies'].append(elm_acc * 100)
        results['elm']['times'].append(elm_time)
        if results['elm']['params'] is None:
            results['elm']['params'] = elm_total
            results['elm']['trainable_params'] = elm_trainable
            results['elm']['frozen_params'] = elm_frozen
        
        print(f"  最佳准确率: {elm_acc*100:.2f}%", flush=True)
        print(f"  训练时间: {elm_time:.2f}秒", flush=True)
        print(f"  可训练参数: {elm_trainable:,} (冻结: {elm_frozen:,})", flush=True)
        print("", flush=True)
    
    # 统计分析
    std_accs = np.array(results['standard']['accuracies'])
    elm_accs = np.array(results['elm']['accuracies'])
    std_times = np.array(results['standard']['times'])
    elm_times = np.array(results['elm']['times'])
    
    t_stat, p_value = stats.ttest_rel(std_accs, elm_accs)
    
    # 打印统计结果
    print("="*70, flush=True)
    print("统计结果", flush=True)
    print("="*70, flush=True)
    print("", flush=True)
    
    print("标准BERT:", flush=True)
    print(f"  平均准确率: {std_accs.mean():.2f}% ± {std_accs.std():.2f}%", flush=True)
    print(f"  最高准确率: {std_accs.max():.2f}%", flush=True)
    print(f"  最低准确率: {std_accs.min():.2f}%", flush=True)
    print(f"  平均训练时间: {std_times.mean():.2f}秒 ± {std_times.std():.2f}秒", flush=True)
    print(f"  总参数: {results['standard']['params']:,}", flush=True)
    print(f"  可训练参数: {results['standard']['trainable_params']:,}", flush=True)
    print("", flush=True)
    
    print("ELM风格BERT (冻结Q,K):", flush=True)
    print(f"  平均准确率: {elm_accs.mean():.2f}% ± {elm_accs.std():.2f}%", flush=True)
    print(f"  最高准确率: {elm_accs.max():.2f}%", flush=True)
    print(f"  最低准确率: {elm_accs.min():.2f}%", flush=True)
    print(f"  平均训练时间: {elm_times.mean():.2f}秒 ± {elm_times.std():.2f}秒", flush=True)
    print(f"  总参数: {results['elm']['params']:,}", flush=True)
    print(f"  可训练参数: {results['elm']['trainable_params']:,}", flush=True)
    print(f"  冻结参数: {results['elm']['frozen_params']:,}", flush=True)
    print("", flush=True)
    
    param_saving = (results['standard']['trainable_params'] - results['elm']['trainable_params']) / results['standard']['trainable_params'] * 100
    acc_diff = elm_accs.mean() - std_accs.mean()
    time_diff = elm_times.mean() - std_times.mean()
    time_percent = time_diff / std_times.mean() * 100
    
    print(f"参数节省: {param_saving:.2f}%", flush=True)
    print(f"准确率差异: {acc_diff:+.2f}%", flush=True)
    print(f"训练时间差异: {time_diff:+.2f}秒 ({time_percent:+.2f}%)", flush=True)
    print("", flush=True)
    
    print("配对t检验:", flush=True)
    print(f"  t统计量: {t_stat:.4f}", flush=True)
    print(f"  p值: {p_value:.4f}", flush=True)
    print(f"  结论: {'差异显著' if p_value < 0.05 else '差异不显著'} (p {'<' if p_value < 0.05 else '>='} 0.05)", flush=True)
    print("", flush=True)
    
    # 保存结果
    output_file = 'bert_imdb_experiments_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"详细结果已保存到: {output_file}", flush=True)
    print("", flush=True)
    
    # 打印详细对比表
    print("="*100, flush=True)
    print("每次实验详细对比", flush=True)
    print("="*100, flush=True)
    print(f"{'实验':<8}{'标准准确率':<15}{'标准时间':<15}{'ELM准确率':<15}{'ELM时间':<15}{'准确率差异':<15}{'时间差异':<10}", flush=True)
    print("-"*100, flush=True)
    
    for i in range(num_experiments):
        acc_diff_i = elm_accs[i] - std_accs[i]
        time_diff_i = elm_times[i] - std_times[i]
        print(f"{i+1:<8}{std_accs[i]:.2f}%{'':<9}{std_times[i]:.1f}s{'':<6}"
              f"{elm_accs[i]:.2f}%{'':<8}{elm_times[i]:.1f}s{'':<6}"
              f"{acc_diff_i:+.2f}%{'':<9}{time_diff_i:+.1f}s", flush=True)


if __name__ == "__main__":
    main()
