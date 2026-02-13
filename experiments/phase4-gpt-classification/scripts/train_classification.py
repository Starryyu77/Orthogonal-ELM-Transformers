"""
分类任务训练脚本 (Phase 4)

支持:
- IMDB情感分析 (2分类)
- AGNews新闻分类 (4分类)
- XNLI/MNLI自然语言推理 (3分类)

使用方法:
    python train_classification.py \
        --model_type baseline \  # 或 oelm_freeze, oelm_random
        --dataset imdb \
        --output_dir outputs/imdb_baseline
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 直接导入模型文件（处理连字符目录名）
import importlib.util

MODELS_DIR = Path(__file__).parent.parent / "models"

# 加载 GPT classification 模型
spec_gpt = importlib.util.spec_from_file_location(
    "modeling_gpt_classification", MODELS_DIR / "modeling_gpt_classification.py"
)
module_gpt = importlib.util.module_from_spec(spec_gpt)
sys.modules["modeling_gpt_classification"] = module_gpt
spec_gpt.loader.exec_module(module_gpt)
GPTForSequenceClassification = module_gpt.GPTForSequenceClassification
create_gpt_classifier = module_gpt.create_gpt_classifier

# 加载 OELM classification 模型
spec_oelm = importlib.util.spec_from_file_location(
    "modeling_oelm_classification", MODELS_DIR / "modeling_oelm_classification.py"
)
module_oelm = importlib.util.module_from_spec(spec_oelm)
sys.modules["modeling_oelm_classification"] = module_oelm
spec_oelm.loader.exec_module(module_oelm)
OELMForSequenceClassification = module_oelm.OELMForSequenceClassification
create_oelm_classifier = module_oelm.create_oelm_classifier


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def get_tokenizer():
    """加载GPT-2 tokenizer。"""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_imdb_dataloader(batch_size: int = 16, max_seq_len: int = 512, split: str = 'train'):
    """加载IMDB数据集。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("需要安装: pip install datasets")
        raise

    logger.info(f"加载IMDB数据集 ({split} split)...")

    tokenizer = get_tokenizer()
    dataset = load_dataset('imdb', split=split)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_len
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=2
    )

    return dataloader, tokenizer


def get_agnews_dataloader(batch_size: int = 16, max_seq_len: int = 512, split: str = 'train'):
    """加载AG News数据集 (4分类新闻)。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("需要安装: pip install datasets")
        raise

    logger.info(f"加载AG News数据集 ({split} split)...")

    tokenizer = get_tokenizer()
    # AG News: 120K训练, 7.6K测试, 4分类
    dataset = load_dataset('ag_news', split=split)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_len
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=2
    )

    return dataloader, tokenizer


def get_xnli_dataloader(batch_size: int = 16, max_seq_len: int = 512, split: str = 'train'):
    """加载XNLI数据集 (3分类NLI)。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("需要安装: pip install datasets")
        raise

    logger.info(f"加载XNLI-en数据集 ({split} split)...")

    tokenizer = get_tokenizer()
    # XNLI英语版: 392K训练, 2.5K验证, 3分类
    if split == 'train':
        dataset = load_dataset('xnli', 'en', split='train')
    else:
        dataset = load_dataset('xnli', 'en', split='validation')

    def tokenize_function(examples):
        # XNLI有premise和hypothesis两个句子
        text = [f"{p} [SEP] {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]
        return tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_seq_len
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=2
    )

    return dataloader, tokenizer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100
):
    """训练一个epoch。"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    step_times = []

    for step, batch in enumerate(dataloader):
        step_start = time.perf_counter()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        loss = model(input_ids, attention_mask=attention_mask, labels=labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()

        # 计算准确率
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

        # 计时
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        if step % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            accuracy = total_correct / total_samples * 100
            logger.info(
                f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:.2f}% | Step time: {step_time*1000:.2f}ms"
            )

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples * 100
    avg_step_time = sum(step_times) / len(step_times)

    return avg_loss, avg_accuracy, avg_step_time


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
):
    """评估模型。"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()

        logits = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='训练GPT分类模型')

    # 模型配置
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['baseline', 'oelm_freeze', 'oelm_random'],
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数')
    parser.add_argument('--vocab_size', type=int, default=50257,
                        help='词表大小')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    # 训练配置
    parser.add_argument('--dataset', type=str, default='imdb',
                        choices=['imdb', 'ag_news', 'xnli', 'mnli'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # 其他
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"加载数据集: {args.dataset}")
    if args.dataset == 'imdb':
        train_loader, tokenizer = get_imdb_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            split='train'
        )
        test_loader, _ = get_imdb_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            split='test'
        )
    elif args.dataset == 'ag_news':
        train_loader, tokenizer = get_agnews_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            split='train'
        )
        test_loader, _ = get_agnews_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            split='test'
        )
    elif args.dataset == 'xnli':
        train_loader, tokenizer = get_xnli_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            split='train'
        )
        test_loader, _ = get_xnli_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            split='validation'
        )
    else:
        raise NotImplementedError(f"数据集 {args.dataset} 暂未实现")

    # 创建模型
    logger.info(f"创建模型: {args.model_type}")
    if args.model_type == 'baseline':
        model = create_gpt_classifier(
            num_classes=args.num_classes,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout
        )
    else:  # oelm_freeze or oelm_random
        init_method = 'orthogonal' if args.model_type == 'oelm_freeze' else 'normal'
        model = create_oelm_classifier(
            num_classes=args.num_classes,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            freeze_qk=True,
            init_method=init_method
        )

    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度
    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate * 0.1
    )

    # TensorBoard (optional)
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / 'logs')
    else:
        writer = None
        logger.info("TensorBoard not available, skipping logging")

    # 训练
    logger.info("开始训练...")
    start_time = time.perf_counter()

    best_accuracy = 0
    timing_stats = {
        'epoch_times': [],
        'step_times': [],
    }

    for epoch in range(args.num_epochs):
        epoch_start = time.perf_counter()

        train_loss, train_acc, avg_step_time = train_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval
        )

        epoch_time = time.perf_counter() - epoch_start
        timing_stats['epoch_times'].append(epoch_time)
        timing_stats['step_times'].append(avg_step_time)

        # 评估
        val_loss, val_acc = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch} 完成 | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'config': vars(args)
            }, output_dir / 'best.pt')
            logger.info(f"保存最佳模型，准确率: {val_acc:.2f}%")

        scheduler.step()

    total_time = time.perf_counter() - start_time

    # 保存最终模型
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'accuracy': val_acc,
        'config': vars(args)
    }, output_dir / 'latest.pt')

    # 保存结果
    results = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'best_accuracy': best_accuracy,
        'final_val_accuracy': val_acc,
        'final_val_loss': val_loss,
        'total_time_seconds': total_time,
        'total_time_formatted': f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s",
        'avg_epoch_time': sum(timing_stats['epoch_times']) / len(timing_stats['epoch_times']),
        'avg_step_time': sum(timing_stats['step_times']) / len(timing_stats['step_times']),
        'config': vars(args)
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'timing_stats.json', 'w') as f:
        json.dump(timing_stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳验证准确率: {best_accuracy:.2f}%")
    logger.info(f"总训练时间: {results['total_time_formatted']}")
    logger.info(f"结果保存在: {output_dir}")
    logger.info("=" * 60)

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
