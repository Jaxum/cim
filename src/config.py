"""
因果实体解析(CIM)框架的配置模块。
本模块作为项目的"中央控制面板"，
定义所有可配置参数和数据集元数据。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class MainConfig:
    """
    由命令行参数直接控制的全局配置参数。
    """
    dataset_name: str = "WDC_Computers"  # 要处理的数据集名称
    run_mode: str = "train"  # 运行模式: "train" 或 "evaluate"
    output_dir: str = "./saved_models"  # 保存模型检查点的目录
    data_dir: str = "./data"  # 数据集根目录
    checkpoint_path: Optional[str] = None  # 评估模式的检查点路径
    oversample_train: bool = True  # 是否在训练期间对少数类进行过采样（对不平衡数据启用）


@dataclass
class ModelConfig:
    """
    模型架构和训练超参数。
    """
    # 编码器配置
    encoder_model_name: str = "d:/Edgedownload/roberta-base"  # 预训练模型路径或Hugging Face模型名
    max_len: int = 128  # 分词器的最大序列长度

    
    # 训练超参数
    epochs: int = 15
    lr: float = 3e-5  
    train_batch_size: int = 8
    eval_batch_size: int = 16
    warmup_steps: int = 20
    weight_decay: float = 0.01
    
    # 因果干预训练(CIT)损失权重
    lambda1: float = 1  # 因果损失权重，防止过度推向负类
    lambda2: float = 0.1  # 混淆损失权重(一致性奖励)
    
    # Focal Loss 可选设置（用于极度不平衡数据）
    focal_gamma: float = 2        # 聚焦参数，γ 越大，模型对易分类样本的关注度越低
    focal_alpha: float = 0.5       #跑完改成0.5（1：1）试试
    
    # 其他训练设置
    seed: int = 123
    gradient_accumulation_steps: int = 8  # 减少梯度累积，加速更新
    max_grad_norm: float = 1.0

    #对比学习
    enable_cit: bool = True  # 是否启用 CIT 干预分支（关闭时不计算 Push/Pull，也不做干预前向）
    causal_margin: float = 0.5
    confounder_margin: float = 0.9
    
    # 阈值搜索配置
    threshold: bool = True  # 是否启用阈值搜索
    threshold_search_start: float = 0.1  # 搜索起始阈值
    threshold_search_end: float = 0.9    # 搜索结束阈值
    threshold_search_step: float = 0.01  # 搜索步长
    
    # 混合精度训练
    use_fp16: bool = True  # 是否使用FP16混合精度训练
    rdrop_alpha: float = 1.0

    # 输入格式：是否为每个属性值添加属性名前缀（对数值/短字段更友好）
    use_attribute_prefix: bool = False

    # 门控权重追踪（按 epoch 保存）
    gate_track_enabled: bool = False  # 是否开启门控权重追踪
    gate_track_scope: str = "all"  # all 或 causal_conf（仅因果+混淆）
    gate_track_out_dir: str = "./gate_traces"  # 输出根目录
    gate_track_max_batches: int = 0  # 0 表示全量统计；>0 表示每轮最多统计的 batch 数



@dataclass
class DatasetConfig:
    """
    数据集特定的元数据和配置。
    定义每个数据集的结构和因果知识。
    """
    name: str  # 数据集名称
    attributes: List[str]  # 要考虑的所有属性列
    causal_attributes: List[str]  # 具有强因果信号的属性
    confounder_attributes: List[str]  # 具有弱信号/混淆信号的属性
    data_path: str  # 数据集目录的相对路径
    label_column: str = "label"  # 标签列名称
    id_columns: List[str] = field(default_factory=lambda: ["ltable_id", "rtable_id"])


# 数据集元数据注册表 - 所有数据集的"知识库"
DATASET_METADATA: Dict[str, DatasetConfig] = {
    

    
    # ER-Magellan结构化数据集

    # 1. Amazon-Google (软件/电子)
    "Magellan_Amazon_Google": DatasetConfig(
        name="Magellan_Amazon_Google",
        data_path="er_magellan/Structured/Amazon-Google",
        attributes=["title", "manufacturer", "price"],
        causal_attributes=["title"],
        confounder_attributes=["manufacturer", "price"],
    ),  

    # 2. Beer
    "Magellan_Beer": DatasetConfig(
        name="Magellan_Beer",
        data_path="er_magellan/Structured/Beer",
        attributes=["Beer_Name", "Brew_Factory_Name", "Style", "ABV"],
        causal_attributes=["Beer_Name", "ABV"], 
        confounder_attributes=["Brew_Factory_Name", "Style"],
    ),  

    # 3. DBLP-ACM 
    "Magellan_DBLP_ACM": DatasetConfig(
        name="Magellan_DBLP_ACM",
        data_path="er_magellan/Structured/DBLP-ACM",
        attributes=["title", "authors", "venue", "year"],
        causal_attributes=["title", "authors"],  
        confounder_attributes=["venue", "year"],  
    ),
    
    # 4. DBLP-GoogleScholar
    "Magellan_DBLP_GoogleScholar": DatasetConfig(
        name="Magellan_DBLP_GoogleScholar",
        data_path="er_magellan/Structured/DBLP-GoogleScholar",
        attributes=["title", "authors", "venue", "year"],
        causal_attributes=["title", "authors"],
        confounder_attributes=["venue", "year"],
    ),
    
    
    # 5. iTunes-Amazon (音乐)
    "Magellan_iTunes_Amazon": DatasetConfig(
        name="Magellan_iTunes_Amazon",
        data_path="er_magellan/Structured/iTunes-Amazon",
        attributes=["Song_Name", "Artist_Name", "Album_Name", "Genre", "Price", "CopyRight", "Time", "Released"],
        causal_attributes=["Song_Name", "Artist_Name"],
        confounder_attributes=["Genre", "Price", "Released", "CopyRight", "Album_Name", "Time"],
    ),

    # 6. Walmart-Amazon (电子/日用品)
    "Magellan_Walmart_Amazon": DatasetConfig(
        name="Magellan_Walmart_Amazon",
        data_path="er_magellan/Structured/Walmart-Amazon",
        attributes=["title", "category", "brand", "modelno", "price"],
        causal_attributes=["title", "modelno"],
        confounder_attributes=["brand", "category", "price"],
    ),
    
    # 7. Abt-Buy (电商产品)
    "Magellan_Abt_Buy": DatasetConfig(
        name="Magellan_Abt_Buy",
        data_path="er_magellan/Textual/Abt-Buy",
        attributes=["name", "description", "price"],
        causal_attributes=["name"],
        confounder_attributes=["price"],
    ),
    
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    获取特定数据集的配置。
    
    参数:
        dataset_name: 数据集名称
        
    返回:
        指定数据集的DatasetConfig对象
        
    抛出:
        ValueError: 如果在DATASET_METADATA中找不到dataset_name
    """
    if dataset_name not in DATASET_METADATA:
        available_datasets = ", ".join(DATASET_METADATA.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in metadata. "
            f"Available datasets: {available_datasets}"
        )
    return DATASET_METADATA[dataset_name]


def get_full_config(dataset_name: str, **overrides) -> tuple:
    """
    获取给定数据集的所有三个配置对象。
    
    参数:
        dataset_name: 数据集名称
        **overrides: 可选的关键字参数，用于覆盖默认值
        
    返回:
        (MainConfig, ModelConfig, DatasetConfig)的元组
    """
    main_config = MainConfig(dataset_name=dataset_name)
    model_config = ModelConfig()
    dataset_config = get_dataset_config(dataset_name)
    
    # 应用任何覆盖值
    for key, value in overrides.items():
        if hasattr(main_config, key):
            setattr(main_config, key, value)
        elif hasattr(model_config, key):
            setattr(model_config, key, value)
    
    return main_config, model_config, dataset_config
