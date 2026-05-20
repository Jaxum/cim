import os
import sys
import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

# 将 src 添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ModelConfig, DatasetConfig, get_dataset_config
from src.dataset import CausalERDataset
from src.model import CIMModel



def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='因果实体解析 (CIM) 框架')
    
    parser.add_argument('--checkpoint_path', type=str, default=None, help='训练好的模型检查点路径')
    parser.add_argument('--dataset_name', type=str, required=True, help='要测试的数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集根目录（默认 ./data）')
    parser.add_argument('--train_size', type=str, choices=['small', 'medium', 'large', 'xlarge'], default='medium', help='WDC 数据集的训练集大小（其他数据集忽略）')
    parser.add_argument('--batch_size', type=int, default=32, help='推理时的批量大小')
    parser.add_argument('--output_file', type=str, default=None, help='保存预测结果的路径（可选）')
    parser.add_argument('--threshold', type=float, default=None, help='分类阈值(0-1)，不指定则默认 0.5')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用的设备')
    
    return parser.parse_args()


def _resolve_split_paths(data_dir: str, dataset_config: DatasetConfig, train_size: str) -> Dict[str, str]:
    """Resolve train/valid/test paths consistently with src/dataset.py."""
    dataset_path = os.path.join(data_dir, dataset_config.data_path)

    if "WDC" in dataset_config.name or "wdc" in dataset_config.data_path.lower():
        train_file = os.path.join(dataset_path, f"train.txt.{train_size}")
        valid_file = os.path.join(dataset_path, f"valid.txt.{train_size}")
        test_file = os.path.join(dataset_path, "test.txt")
    else:
        train_file = os.path.join(dataset_path, "train.txt")
        valid_file = os.path.join(dataset_path, "valid.txt")
        test_file = os.path.join(dataset_path, "test.txt")

    return {"train": train_file, "valid": valid_file, "test": test_file}


def _make_eval_loader(
    file_path: str,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    batch_size: int,
) -> DataLoader:
    dataset = CausalERDataset(
        file_path=file_path,
        dataset_config=dataset_config,
        model_config=model_config,
        mode="test",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )




def load_model(checkpoint_path: str, model_config: ModelConfig, dataset_config: DatasetConfig, device: str) -> CIMModel:
    """
    从检查点加载训练好的模型。
    
    参数:
        checkpoint_path: 模型检查点路径
        model_config: 模型配置
        dataset_config: 数据集配置
        device: 加载模型的设备
        
    返回:
        加载后的 CIMModel
    """
    print(f"加载模型: {checkpoint_path}")
    
    # 初始化模型
    model = CIMModel(model_config, dataset_config)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_result.missing_keys:
            print(f"警告 没找到: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"警告 意外的键: {load_result.unexpected_keys}")
        print(f"从 epoch {checkpoint.get('best_epoch', 'unknown')} 加载模型")
        if 'best_f1' in checkpoint:
            print(f"Checkpoint best F1: {checkpoint['best_f1']:.4f}")
    else:
        load_result = model.load_state_dict(checkpoint, strict=False)
        if load_result.missing_keys:
            print(f"警告 没找到: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"警告 意外的键: {load_result.unexpected_keys}")
    
    model = model.to(device)
    model.eval()
    
    return model


def predict(
    model: CIMModel,
    dataloader: DataLoader,
    device: str,
    batch_size: int,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    在测试数据上运行推理。
    
    参数:
        model: 训练好的 CIMModel
        dataloader: 测试数据的 DataLoader
        device: 运行设备
        batch_size: 用于进度报告的批量大小
        threshold: 分类阈值
        
    返回:
        包含预测结果、标签和门控值的字典
    """
    all_predictions = []
    all_labels = []
    all_probs = []
    all_gate_values = []

    print("\n预测中...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测", total=len(dataloader)):
            # 将批次移至设备
            normal_input_ids = batch["normal_input_ids"].to(device)
            normal_attention_mask = batch["normal_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # 前向传播
            outputs = model(normal_input_ids, normal_attention_mask)
            logits = outputs["logits"]
            gates = outputs["gates"]  # 获取门控值
            
            # 获取预测结果（使用阈值）
            probs = torch.softmax(logits, dim=-1)
            predictions = (probs[:, 1] >= threshold).long()
            
            # 存储结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 匹配概率（类别 1）
            all_gate_values.append(gates.cpu().numpy())  # 存储门控值
    
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'gate_values': np.vstack(all_gate_values)  # 形状: (num_samples, num_attributes)
    }


def main():
    args = parse_arguments()

    dataset_config = get_dataset_config(args.dataset_name)

    # ModelConfig: 优先用 checkpoint 里的（能对齐训练时的 max_len / encoder 等），再允许少量覆盖
    model_config = ModelConfig()

    if args.checkpoint_path is None:
        raise ValueError("必须提供 --checkpoint_path")

    ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and "model_config" in ckpt:
        try:
            model_config = ckpt["model_config"]
        except Exception:
            model_config = ModelConfig()

    # matcher 的 batch_size 以命令行为准
    model_config.eval_batch_size = args.batch_size

    # 构建数据路径与 loader
    paths = _resolve_split_paths(args.data_dir, dataset_config, args.train_size)
    test_loader = _make_eval_loader(paths["test"], dataset_config, model_config, batch_size=args.batch_size)

    # 加载模型
    model = load_model(args.checkpoint_path, model_config, dataset_config, args.device)

    # 决定阈值
    if args.threshold is not None:
        threshold_to_use = float(args.threshold)
    else:
        threshold_to_use = 0.5

    # 在 test 上预测与评估
    results = predict(model, test_loader, args.device, batch_size=args.batch_size, threshold=threshold_to_use)
    metrics = compute_metrics(results["predictions"], results["labels"])

    print(f"\n使用阈值: {threshold_to_use:.3f}")
    print_results(metrics, results, dataset_config)

    if args.output_file:
        save_predictions(results, metrics, args.output_file)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标。
    
    参数:
        predictions: 预测标签
        labels: 真实标签
        
    返回:
        指标字典
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }
    
    return metrics


def print_results(metrics: Dict[str, float], results: Dict[str, Any], dataset_config):
    """打印评估结果，包括门控值分析。"""
    print("\n" + "="*60)
    print("结果")
    print("="*60)
    
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率:    {metrics['recall']:.4f}")
    print(f"F1 分数:  {metrics['f1']:.4f}")
    
    print("\n" + "-"*60)
    print("分类报告:")
    print("-"*60)
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['非匹配', '匹配'],
        digits=4
    ))
    
    # 门控值分析
    print("\n" + "-"*60)
    print("属性门控值分析（因果重要性）:")
    print("-"*60)
    
    gate_values = results['gate_values']  # 形状: (num_samples, num_attributes)
    attributes = dataset_config.attributes
    causal_attrs = set(dataset_config.causal_attributes)
    confounder_attrs = set(dataset_config.confounder_attributes)
    
    print(f"\n{'属性':<20} {'类型':<10} {'平均门控值':<12} {'标准差':<10}")
    print("-" * 60)
    
    for i, attr in enumerate(attributes):
        mean_gate = gate_values[:, i].mean()
        std_gate = gate_values[:, i].std()
        
        if attr in causal_attrs:
            attr_type = "因果"
        elif attr in confounder_attrs:
            attr_type = "混淆"
        else:
            attr_type = "其他"
        
        print(f"{attr:<20} {attr_type:<10} {mean_gate:<12.4f} {std_gate:<10.4f}")
    
    # 因果属性与混淆属性的对比
    if causal_attrs and confounder_attrs:
        causal_indices = [i for i, attr in enumerate(attributes) if attr in causal_attrs]
        conf_indices = [i for i, attr in enumerate(attributes) if attr in confounder_attrs]
        
        causal_mean = gate_values[:, causal_indices].mean()
        conf_mean = gate_values[:, conf_indices].mean()
        
        print("\n" + "-"*60)
        print("属性类型对比:")
        print(f"  因果属性平均门控值:   {causal_mean:.4f}")
        print(f"  混淆属性平均门控值:   {conf_mean:.4f}")
        
        if causal_mean > conf_mean:
            print("   因果属性重要性更高（符合预期）")
        else:
            print("   混淆属性重要性更高（需要检查）")
    
    # 额外统计信息
    total_samples = len(results['labels'])
    num_matches = np.sum(results['labels'] == 1)
    num_non_matches = np.sum(results['labels'] == 0)
    
    print("-"*60)
    print("数据集统计:")
    print(f"  样本总数: {total_samples}")
    print(f"  匹配数: {num_matches} ({num_matches/total_samples*100:.2f}%)")
    print(f"  非匹配数: {num_non_matches} ({num_non_matches/total_samples*100:.2f}%)")
    print("="*60)


def save_predictions(
    results: Dict[str, Any],
    metrics: Dict[str, float],
    output_file: str
):
    """
    将预测结果保存到文件。
    
    参数:
        results: 预测结果
        metrics: 计算出的指标
        output_file: 输出文件路径
    """
    output_data = {
        'metrics': metrics,
        'predictions': results['predictions'].tolist(),
        'labels': results['labels'].tolist(),
        'probs': results['probs'].tolist()
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n预测结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
