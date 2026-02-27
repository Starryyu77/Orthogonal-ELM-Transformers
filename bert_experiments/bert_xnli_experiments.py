#!/usr/bin/env python3
"""
BERT XNLI实验：标准Fine-tuning vs ELM风格Fine-tuning（冻结Q,K）
使用预训练BERT-base模型在XNLI任务上进行对比实验
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
import argparse
from scipy import stats
import numpy as np
from tqdm import tqdm

try:
    from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
    from torch.optim import AdamW
    from datasets import load_dataset
except ImportError:
    print("错误：需要安装transformers和datasets库")
    print("运行: pip install transformers datasets")
    exit(1)


class XNLIBertDataset(Dataset):
    """XNLI数据集（使用BERT tokenizer）"""
    def __init__(self, premises, hypotheses, labels, tokenizer, max_len=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, idx):
        premise = str(self.premises[idx])
        hypothesis = str(self.hypotheses[idx])
        label = self.labels[idx]
        
        # BERT tokenization
        encoding = self.tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_xnli_data(split='train', max_samples=None, language='en'):
    """加载XNLI数据集"""
    print(f"加载XNLI {split}集（语言: {language}）...", flush=True)
    
    if split == 'train':
        dataset = load_dataset('xnli', language, split='train')
    elif split == 'validation':
        dataset = load_dataset('xnli', language, split='validation')
    elif split == 'test':
        dataset = load_dataset('xnli', language, split='test')
    else:
        raise ValueError(f"Unknown split: {split}")
    
    premises = []
    hypotheses = []
    labels = []
    
    for example in dataset:
        premises.append(example['premise'])
        hypotheses.append(example['hypothesis'])
        labels.append(example['label'])  # 0: entailment, 1: neutral, 2: contradiction
        
        if max_samples and len(premises) >= max_samples:
            break
    
    return premises, hypotheses, labels


def freeze_bert_qk_projections(model):
    """冻结BERT所有attention层的Q,K投影"""
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 冻结所有attention层的query和key投影
        if 'attention.self.query' in name or 'attention.self.key' in name:
            param.requires_grad = False
            frozen_params += param.numel()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数: {total_params:,}", flush=True)
    print(f"  冻结参数: {frozen_params:,}", flush=True)
    print(f"  可训练参数: {trainable_params:,}", flush=True)
    print(f"  参数节省: {frozen_params/total_params*100:.2f}%", flush=True)
    
    return trainable_params


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def run_single_experiment(tokenizer, train_loader, test_loader, device, seed, 
                         freeze_qk=False, num_epochs=3, learning_rate=2e-5):
    """运行单次实验"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 加载预训练BERT
    print(f"  加载BERT模型...", flush=True)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    ).to(device)
    
    # ELM风格：冻结Q,K投影
    if freeze_qk:
        trainable_params = freeze_bert_qk_projections(model)
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  可训练参数: {trainable_params:,}", flush=True)
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
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
    
    return best_acc, training_time, trainable_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='基础随机种子')
    parser.add_argument('--num-exp', type=int, default=10, help='实验次数')
    parser.add_argument('--train-samples', type=int, default=None, help='训练样本数（None表示全部）')
    parser.add_argument('--test-samples', type=int, default=None, help='测试样本数（None表示全部）')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--language', type=str, default='en', help='XNLI语言（en, fr, de等）')
    args = parser.parse_args()
    
    base_seed = args.seed
    num_experiments = args.num_exp
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}", flush=True)
    
    print("", flush=True)
    print("="*70, flush=True)
    print(f"BERT XNLI实验：标准Fine-tuning vs ELM风格Fine-tuning（冻结Q,K）", flush=True)
    print(f"数据集: XNLI (Cross-lingual NLI, 语言: {args.language})", flush=True)
    print(f"实验次数: {num_experiments}", flush=True)
    print(f"基础随机种子: {base_seed}", flush=True)
    print(f"训练轮数: {args.epochs}", flush=True)
    print(f"批次大小: {args.batch_size}", flush=True)
    print("="*70, flush=True)
    print("", flush=True)
    
    # 加载数据
    print(f"加载XNLI数据集（语言: {args.language}）...", flush=True)
    train_premises, train_hypotheses, train_labels = load_xnli_data('train', max_samples=args.train_samples, language=args.language)
    test_premises, test_hypotheses, test_labels = load_xnli_data('validation', max_samples=args.test_samples, language=args.language)
    
    print(f"训练样本: {len(train_premises)}", flush=True)
    print(f"测试样本: {len(test_premises)}", flush=True)
    print("", flush=True)
    
    # 初始化tokenizer
    print("加载BERT tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("", flush=True)
    
    results = {
        'standard': {'accuracies': [], 'times': [], 'params': None},
        'elm': {'accuracies': [], 'times': [], 'params': None}
    }
    
    for exp_idx in range(num_experiments):
        seed = base_seed + exp_idx * 100
        
        print("="*70, flush=True)
        print(f"实验 {exp_idx + 1}/{num_experiments} (seed={seed})", flush=True)
        print("="*70, flush=True)
        print("", flush=True)
        
        # 准备数据加载器
        train_dataset = XNLIBertDataset(train_premises, train_hypotheses, train_labels, tokenizer)
        test_dataset = XNLIBertDataset(test_premises, test_hypotheses, test_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 标准BERT
        print("测试 标准BERT Fine-tuning...", flush=True)
        std_acc, std_time, std_params = run_single_experiment(
            tokenizer, train_loader, test_loader, device, seed,
            freeze_qk=False, num_epochs=args.epochs, learning_rate=args.lr
        )
        results['standard']['accuracies'].append(std_acc * 100)
        results['standard']['times'].append(std_time)
        if results['standard']['params'] is None:
            results['standard']['params'] = std_params
        
        print(f"  最佳准确率: {std_acc*100:.2f}%", flush=True)
        print(f"  训练时间: {std_time:.2f}秒", flush=True)
        print("", flush=True)
        
        # ELM-BERT
        print("测试 ELM-BERT Fine-tuning (冻结Q,K)...", flush=True)
        elm_acc, elm_time, elm_params = run_single_experiment(
            tokenizer, train_loader, test_loader, device, seed,
            freeze_qk=True, num_epochs=args.epochs, learning_rate=args.lr
        )
        results['elm']['accuracies'].append(elm_acc * 100)
        results['elm']['times'].append(elm_time)
        if results['elm']['params'] is None:
            results['elm']['params'] = std_params
            results['elm']['trainable_params'] = elm_params
            results['elm']['frozen_params'] = std_params - elm_params
        
        print(f"  最佳准确率: {elm_acc*100:.2f}%", flush=True)
        print(f"  训练时间: {elm_time:.2f}秒", flush=True)
        print("", flush=True)
        
        print(f"对比 (实验 {exp_idx + 1}):", flush=True)
        print(f"  准确率差异: {(elm_acc - std_acc)*100:+.2f}%", flush=True)
        print(f"  时间差异: {elm_time - std_time:+.2f}秒", flush=True)
        print("", flush=True)
    
    # 统计分析
    std_accs = np.array(results['standard']['accuracies'])
    elm_accs = np.array(results['elm']['accuracies'])
    std_times = np.array(results['standard']['times'])
    elm_times = np.array(results['elm']['times'])
    
    print("="*70, flush=True)
    print("统计结果", flush=True)
    print("="*70, flush=True)
    print("", flush=True)
    
    print("准确率 (%):", flush=True)
    print(f"  标准BERT: {std_accs.mean():.2f} ± {std_accs.std():.2f}", flush=True)
    print(f"  ELM-BERT: {elm_accs.mean():.2f} ± {elm_accs.std():.2f}", flush=True)
    print(f"  差异: {(elm_accs - std_accs).mean():+.2f}%", flush=True)
    print("", flush=True)
    
    print("训练时间 (秒):", flush=True)
    print(f"  标准BERT: {std_times.mean():.2f} ± {std_times.std():.2f}", flush=True)
    print(f"  ELM-BERT: {elm_times.mean():.2f} ± {elm_times.std():.2f}", flush=True)
    print(f"  加速: {(1 - elm_times.mean()/std_times.mean())*100:.2f}%", flush=True)
    print("", flush=True)
    
    print("参数统计:", flush=True)
    print(f"  总参数: {results['standard']['params']:,}", flush=True)
    print(f"  冻结参数: {results['elm']['frozen_params']:,}", flush=True)
    print(f"  参数节省: {results['elm']['frozen_params']/results['standard']['params']*100:.2f}%", flush=True)
    print("", flush=True)
    
    # t检验
    t_stat, p_value = stats.ttest_rel(std_accs, elm_accs)
    print("配对t检验:", flush=True)
    print(f"  t统计量: {t_stat:.4f}", flush=True)
    print(f"  p值: {p_value:.4f}", flush=True)
    print(f"  结论: {'差异显著' if p_value < 0.05 else '差异不显著'} (p {'<' if p_value < 0.05 else '>='} 0.05)", flush=True)
    print("", flush=True)
    
    output_file = f'bert_xnli_{args.language}_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"详细结果已保存到: {output_file}", flush=True)
    print("", flush=True)
    
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
