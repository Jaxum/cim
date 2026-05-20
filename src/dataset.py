"""
因果实体解析（CIM）框架的数据集模块。
本模块充当"数据厨师"，负责：
- 从 CSV 文件加载原始数据
- 创建干预样本（因果属性和混淆属性）
- 对文本数据进行分词
- 准备批量数据加载器
"""

import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional
import numpy as np

from .config import MainConfig, ModelConfig, DatasetConfig


class CausalERDataset(Dataset):
    """
    因果实体解析的 PyTorch 数据集。
    
    为每个数据行生成三种类型的样本：
    1. 正常样本：原始实体对
    2. 因果干预样本：核心因果属性被修改
    3. 混淆干预样本：混淆属性被修改
    """
    
    def __init__(
        self,
        file_path: str,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        mode: str = "train"
    ):
        """
        初始化数据集。
        
        Args:
            file_path: 数据文件路径（CSV 或 TXT）
            dataset_config: 数据集特定配置
            model_config: 模型配置
            mode: "train"、"valid" 或 "test"
        """
        self.file_path = file_path
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.mode = mode
        
        # 加载数据
        self.data = self._load_data()
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.encoder_model_name
        )
        
        # 构建用于生成干预样本的干扰词池
        self.enable_cit = getattr(self.model_config, "enable_cit", True)
        if self.mode == "train" and self.enable_cit:
            self.distractor_pool = self._build_distractor_pool()
        else:
            self.distractor_pool = {}
        
    
    def _load_data(self) -> pd.DataFrame:
        """
        从文件加载数据。
        
        Returns:
            包含加载数据的 DataFrame
        """
        # 判断文件格式
        if '.txt' in self.file_path:
            df = self._parse_custom_format(self.file_path)
        else:
            raise ValueError(f"不支持的文件格式: {self.file_path}")
        
        # 用空字符串填充 NaN 值
        df = df.fillna("")
        
        return df
    
    def _parse_custom_format(self, file_path: str) -> pd.DataFrame:
        """
        解析 txt 文件中使用的自定义 "COL name VAL value" 格式。
        
        Args:
            file_path: txt 文件路径
            
        Returns:
            正确解析列的 DataFrame
        """
        import re
        
        data_rows = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:  # 至少需要：left_entity, right_entity, label
                    continue
                
                row = {}
                
                # 解析左实体（第一部分）
                left_text = parts[0]
                for attr in self.dataset_config.attributes:
                    pattern = f'COL {attr} VAL (.*?)(?:COL |$)'
                    match = re.search(pattern, left_text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip().strip('"').strip("'")
                        row[f'ltable_{attr}'] = value
                    else:
                        row[f'ltable_{attr}'] = ""
                
                # 解析右实体（第二部分）
                right_text = parts[1]
                for attr in self.dataset_config.attributes:
                    pattern = f'COL {attr} VAL (.*?)(?:COL |$)'
                    match = re.search(pattern, right_text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip().strip('"').strip("'")
                        row[f'rtable_{attr}'] = value
                    else:
                        row[f'rtable_{attr}'] = ""
                
                # 解析标签（最后部分）
                row[self.dataset_config.label_column] = int(parts[-1].strip())
                
                data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def _build_distractor_pool(self) -> Dict[str, List[str]]:
        """
        为每个属性构建干扰值池。
        该池用于生成干预样本。
        
        Returns:
            将属性名映射到唯一值列表的字典
        """
        distractor_pool = {}
        
        for attr in self.dataset_config.attributes:
            # 从左表和右表收集值
            ltable_col = f"ltable_{attr}"
            rtable_col = f"rtable_{attr}"
            
            values = set()
            
            if ltable_col in self.data.columns:
                values.update(self.data[ltable_col].dropna().astype(str).unique())
            
            if rtable_col in self.data.columns:
                values.update(self.data[rtable_col].dropna().astype(str).unique())
            
            # 删除空字符串
            values.discard("")
            
            distractor_pool[attr] = list(values)
        
        return distractor_pool
    
    def _create_intervention(
        self,
        row: Dict[str, Any],
        attributes_to_modify: List[str]
    ) -> Dict[str, Any]:
        
        """
        对数据进行因果或混淆干预。
        
        Args:
            row: 原始数据行（字典形式）
            attributes_to_modify: 要修改的属性名列表
            
        Returns:
            修改后的数据行
        """
                
        import re 
        import random

        modified_row = row.copy()
        
        for attr in attributes_to_modify:
            ltable_col = f"ltable_{attr}"
            rtable_col = f"rtable_{attr}"
            
            # 随机决定改哪一边 
            target_side = "left" if random.random() < 0.5 else "right"
            target_col = ltable_col if target_side == "left" else rtable_col
            
            # 如果该列不存在，跳过
            if target_col not in row:
                continue

            current_val = str(row.get(target_col, ""))
            
            # 简单的正则判断是否包含数字
            # 若包含数字，则进行概率干预
            has_digit = bool(re.search(r'\d', current_val))
            
            intervention_applied = False
            
            if has_digit and random.random() < 0.7: # 70% 概率做数值微调
                def replace_num(match):
                    val = match.group()
                    try:
                        if '.' in val: # 价格/浮点数
                            # 变成 0.1-2.5 倍的随机数
                            new_val = float(val) * random.choice([0.1, 1.9, 1.5, 0.5, 2.5]) 
                            return f"{new_val:.2f}"
                        else: # 版本号/整数
                            # +1, -1, 变年份等
                            new_val = int(val) + random.choice([1, -1, 10, -10])
                            return str(max(0, new_val))
                    except:
                        return val
                
                # 尝试替换
                new_value = re.sub(r'\d+(\.\d+)?', replace_num, current_val, count=1)
                
                if new_value != current_val:
                    modified_row[target_col] = new_value
                    intervention_applied = True
            
            # 如果没做数值干预，则做基于 Pool 的随机替换
            if not intervention_applied:
                pool = self.distractor_pool.get(attr, [])
                if not pool:
                    continue
                
                # 获取两边的当前值，确保新采样值与两边都不同
                val_left = str(row.get(ltable_col, ""))
                val_right = str(row.get(rtable_col, ""))
                
                attempts = 0
                while attempts < 10:
                    new_value = random.choice(pool)
                    # 关键：干扰项不能等于左边，也不能等于右边
                    if new_value != val_left and new_value != val_right:
                        break
                    attempts += 1
                
                modified_row[target_col] = new_value
        
        return modified_row
    
    def _tokenize_row(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        对每行数据进行分词。
        
        Args:
            row: 数据行（字典形式）
            
        Returns:
            包含每个属性对的分词张量的字典
        """
        # 存储所有分词后的属性对
        all_input_ids = []
        all_attention_masks = []
        
        for attr in self.dataset_config.attributes:
            ltable_col = f"ltable_{attr}"
            rtable_col = f"rtable_{attr}"
            
            # 获取属性值
            left_value = str(row.get(ltable_col, ""))
            right_value = str(row.get(rtable_col, ""))

            # 可选：加上属性名前缀，让编码器知道“这是 price / modelno / title ...”
            # 对短字段/数字字段尤其有帮助。
            if getattr(self.model_config, "use_attribute_prefix", False):
                left_value = f"{attr}: {left_value}".strip()
                right_value = f"{attr}: {right_value}".strip()
            
            # 使用分词器自动处理句子对
            # 这将自动添加正确的特殊标记
            encoded = self.tokenizer(
                left_value,
                right_value,
                max_length=self.model_config.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_attention_masks.append(encoded["attention_mask"].squeeze(0))
        
        # 堆叠所有属性对
        # 形状: (num_attributes, max_len)
        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks)
        }
    
    def __len__(self) -> int:
        """返回数据集中的样本数量。"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本及其因果干预和混淆干预样本。
        
        Args:
            idx: 样本索引
            
        Returns:
            包含以下内容的字典：
            - normal_input_ids, normal_attention_mask: 原始样本
            - causal_input_ids, causal_attention_mask: 因果干预样本
            - conf_input_ids, conf_attention_mask: 混淆干预样本
            - label: 真实标签
            - is_match: 是否为正匹配
        """
        # 将行转换为字典，并规范键类型为 str（便于静态类型检查）
        row_raw = self.data.iloc[idx].to_dict()
        row: Dict[str, Any] = {str(k): v for k, v in row_raw.items()}
        
        # 获取标签
        label = int(row[self.dataset_config.label_column])
        
        # 对正常样本进行分词
        normal_tokens = self._tokenize_row(row)
        
        result = {
            "normal_input_ids": normal_tokens["input_ids"],
            "normal_attention_mask": normal_tokens["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
            "is_match": torch.tensor(label == 1, dtype=torch.bool)
        }
        
        # 仅在训练集、正匹配且启用 CIT 时生成干预样本
        if self.mode == "train" and label == 1 and self.enable_cit:
            # 创建因果干预样本（如果存在因果属性）
            if self.dataset_config.causal_attributes:
                causal_row = self._create_intervention(
                    row,
                    self.dataset_config.causal_attributes
                )
                causal_tokens = self._tokenize_row(causal_row)
                result["causal_input_ids"] = causal_tokens["input_ids"]
                result["causal_attention_mask"] = causal_tokens["attention_mask"]
            else:
                # 没有因果属性，使用虚拟张量
                num_attrs = len(self.dataset_config.attributes)
                max_len = self.model_config.max_len
                result["causal_input_ids"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
                result["causal_attention_mask"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
            
            # 创建混淆干预样本（仅当存在混淆属性时）
            if self.dataset_config.confounder_attributes:
                conf_row = self._create_intervention(
                    row,
                    self.dataset_config.confounder_attributes
                )
                conf_tokens = self._tokenize_row(conf_row)
                result["conf_input_ids"] = conf_tokens["input_ids"]
                result["conf_attention_mask"] = conf_tokens["attention_mask"]
            else:
                # 没有混淆属性，使用虚拟张量
                num_attrs = len(self.dataset_config.attributes)
                max_len = self.model_config.max_len
                result["conf_input_ids"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
                result["conf_attention_mask"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
        else:
            # 对于非训练样本或负匹配，创建虚拟张量
            num_attrs = len(self.dataset_config.attributes)
            max_len = self.model_config.max_len
            
            result["causal_input_ids"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
            result["causal_attention_mask"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
            result["conf_input_ids"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
            result["conf_attention_mask"] = torch.zeros(num_attrs, max_len, dtype=torch.long)
        
        return result


def create_dataloader(
    file_path: str,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    mode: str = "train",
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    创建 DataLoader 
    
    Args:
        file_path: 数据文件路径
        dataset_config: 数据集配置
        model_config: 模型配置
        mode: "train"、"valid" 或 "test"
        shuffle: 是否打乱数据（默认：训练时为 True，否则为 False）
        
    Returns:
        PyTorch DataLoader 实例
    """
    # 创建数据集
    dataset = CausalERDataset(
        file_path=file_path,
        dataset_config=dataset_config,
        model_config=model_config,
        mode=mode
    )
    
    # 确定批次大小和是否打乱
    batch_size = (
        model_config.train_batch_size if mode == "train"
        else model_config.eval_batch_size
    )
    
    if shuffle is None:
        shuffle = (mode == "train")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows下建议设为0，避免多进程开销和死锁
        pin_memory=True,  # 启用锁页内存加速GPU传输
    )
    
    return dataloader


def get_dataloaders(
    main_config: MainConfig,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    train_size: str = "medium"
) -> tuple:
    """
    为数据集创建训练、验证和测试数据加载器。
    
    Args:
        main_config: 主配置
        model_config: 模型配置
        dataset_config: 数据集配置
        train_size: WDC 数据集的训练集大小（"small"、"medium"、"large"、"xlarge"）
        
    Returns:
        (train_loader, valid_loader, test_loader) 的元组
    """
    data_dir = main_config.data_dir
    dataset_path = os.path.join(data_dir, dataset_config.data_path)
    
    # 根据数据集类型确定文件名
    if "WDC" in dataset_config.name or "wdc" in dataset_config.data_path.lower():
        # WDC 数据集有特定大小的训练文件
        train_file = os.path.join(dataset_path, f"train.txt.{train_size}")
        valid_file = os.path.join(dataset_path, f"valid.txt.{train_size}")
        test_file = os.path.join(dataset_path, "test.txt")
    else:
        # 其他数据集有简单的 train/valid/test 文件
        train_file = os.path.join(dataset_path, "train.txt")
        valid_file = os.path.join(dataset_path, "valid.txt")
        test_file = os.path.join(dataset_path, "test.txt")
    
    # 创建数据加载器
    # 对于训练加载器，如果开启 oversample_train，则使用 WeightedRandomSampler
    train_dataset = CausalERDataset(
        file_path=train_file,
        dataset_config=dataset_config,
        model_config=model_config,
        mode="train"
    )

    if main_config.oversample_train:
        # 计算每个样本的权重: 逆频率
        labels = train_dataset.data[dataset_config.label_column].astype(int).values
        labels_array = np.asarray(labels)  # 显式转为 numpy array
        unique, counts = np.unique(labels_array, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))
        # 权重 per class: 1 / count
        class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
        sample_weights = np.array([class_weights[int(l)] for l in labels], dtype=np.float32)
        # 转换为 PyTorch 张量以兼容 WeightedRandomSampler 的类型检查
        sample_weights_tensor = torch.from_numpy(sample_weights).float()
        # 使用 Python list 来消除类型检查差异
        sampler = WeightedRandomSampler(weights=sample_weights_tensor.tolist(), num_samples=len(sample_weights_tensor), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.train_batch_size,
            sampler=sampler,
            num_workers=0, # Windows下建议设为0
            pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = create_dataloader(
            train_file, dataset_config, model_config, mode="train", shuffle=True
        )
    
    valid_loader = create_dataloader(
        valid_file, dataset_config, model_config, mode="valid", shuffle=False
    )
    
    test_loader = create_dataloader(
        test_file, dataset_config, model_config, mode="test", shuffle=False
    )
    
    return train_loader, valid_loader, test_loader
