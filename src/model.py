"""
因果实体解析（CIM）框架的模型模块。
本模块定义了神经网络架构，包括：
- CausalAggregationNetwork (CAN): 属性级因果门控机制
- CIMModel: 集成编码器、CAN 和分类器的完整模型
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple
import torch.nn.functional as F

from .config import ModelConfig, DatasetConfig


class CausalAggregationNetwork(nn.Module):
    """
    因果聚合网络（CAN）- CIM 的第一支柱。
    
    本模块为每个属性实现独立的因果门控，
    在架构上强制解耦：
    1. "证据是什么？" - 属性相似度
    2. "这个证据有多重要？" - 因果重要性（门控值）
    """
    
    def __init__(self, hidden_size: int, num_attributes: int):
        """
        初始化因果聚合网络。
        
        参数:
            hidden_size: 证据向量的维度
            num_attributes: 数据集中的属性数量
        """
        super(CausalAggregationNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attributes = num_attributes
        

        # --- 新增模块：上下文融合层 ---
        # 使用单层 Transformer 让属性之间交互
        # nhead=4 意味着让模型从4个不同角度关注其他属性

        #这不仅仅是加了个 Transformer。这叫 "Context-Aware Causal Inference" (上下文感知的因果推理)。
        #Story: "单一属性的因果效力取决于上下文。例如，'Software' 这个词在 Title 中可能是因果性的（决定类别），
        #但如果 Description 里已经详细描述了功能，Title 里的这个词就变得不重要了。我们的 Context Layer 实现了这种动态调整。"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=4, 
            dim_feedforward=hidden_size*2,
            dropout=0.1,
            batch_first=True  # 关键：输入形状为 (batch, seq, feature)
        )
        self.context_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # ---------------------------

        # 用于残差连接后的归一化
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 为每个属性创建独立的门控网络
        # 每个门控学习确定其属性的因果重要性
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()  # 门控值在 [0, 1] 范围内
            )
            for _ in range(num_attributes)
        ])

        # 门控初始化由 CIMModel 根据配置决定
    
    def forward(self, evidence_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：应用因果门控并聚合证据。
        
        参数:
            evidence_vectors: 形状为 (batch_size, num_attributes, hidden_size) 的张量
                             每个向量代表一个属性对的证据
        
        返回:
            aggregated_vector: 形状为 (batch_size, hidden_size) 的张量
                              因果加权的聚合证据
            gate_values: 形状为 (batch_size, num_attributes) 的张量
                        每个属性学习到的因果重要性
        """

        # --- 新增步骤：属性交互 ---
        # 原始证据向量是独立的，现在我们让它们通过 Self-Attention 融合
        # 这样 Title 就能"看到" Description 里的信息
        context_evidence = self.context_layer(evidence_vectors)
        
        # 残差连接 + 归一化：把原始证据加回去，防止信息丢失
        # 这一步让模型既看"上下文"，也看"原始对齐"
        evidence_vectors = self.layer_norm(evidence_vectors + context_evidence)
        # -----------------------


        batch_size = evidence_vectors.size(0)
        
        # 存储每个属性的门控值
        gate_values = []
        weighted_vectors = []
        
        # 独立处理每个属性
        for i in range(self.num_attributes):
            # 提取属性 i 的证据向量
            # 形状: (batch_size, hidden_size)
            attr_evidence = evidence_vectors[:, i, :]
            
            # 使用属性专用的门控网络计算门控值
            # 形状: (batch_size, 1)
            gate = self.gates[i](attr_evidence)
            gate_values.append(gate)
            
            # 用门控值对证据向量加权
            # 形状: (batch_size, hidden_size)
            weighted = attr_evidence * gate
            weighted_vectors.append(weighted)
        
        # 堆叠门控值: (batch_size, num_attributes)
        gate_values = torch.cat(gate_values, dim=1)
        
        # 堆叠并求和加权向量: (batch_size, hidden_size)
        weighted_vectors = torch.stack(weighted_vectors, dim=1)  # (batch_size, num_attrs, hidden)
        aggregated_vector = torch.sum(weighted_vectors, dim=1)   # (batch_size, hidden)
        
        return aggregated_vector, gate_values


class CIMModel(nn.Module):
    """
    完整的因果实体解析模型。
    
    架构流程：
    1. 使用共享的预训练编码器对每个属性对进行编码
    2. 通过比较左/右表示构建证据向量
    3. 应用因果聚合网络（CAN）
    4. 使用最终线性层进行分类
    """
    
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        """
        初始化 CIM 模型。
        
        参数:
            model_config: 模型配置
            dataset_config: 数据集配置（用于 num_attributes）
        """
        super(CIMModel, self).__init__()
        
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.num_attributes = len(dataset_config.attributes)
        
        # 加载预训练编码器
        self.encoder = AutoModel.from_pretrained(model_config.encoder_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        
        # 因果聚合网络
        self.can = CausalAggregationNetwork(
            hidden_size=self.hidden_size,
            num_attributes=self.num_attributes
        )
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 2)  # 二分类
        )
    
    def encode_attribute_pairs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        使用共享编码器对所有属性对进行编码。
        
        参数:
            input_ids: 形状为 (batch_size, num_attributes, max_len) 的张量
            attention_mask: 形状为 (batch_size, num_attributes, max_len) 的张量
        
        返回:
            形状为 (batch_size, num_attributes, hidden_size) 的张量
            包含每个属性对的 [CLS] 标记表示
        """
        batch_size, num_attrs, max_len = input_ids.size()
        
        # 重塑以并行处理所有属性对
        # (batch_size * num_attributes, max_len)
        input_ids_flat = input_ids.view(-1, max_len)
        attention_mask_flat = attention_mask.view(-1, max_len)
        
        # 使用预训练模型进行编码
        outputs = self.encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        
        # 提取 [CLS] 标记表示（第一个标记）
        # 形状: (batch_size * num_attributes, hidden_size)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 重塑回分离的属性
        # 形状: (batch_size, num_attributes, hidden_size)
        cls_embeddings = cls_embeddings.view(batch_size, num_attrs, self.hidden_size)
        
        return cls_embeddings
    
    def construct_evidence_vectors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        从属性对构建证据向量。
        
        每个属性的证据向量表示该属性的左右实体值之间的相似度/差异。
        
        Args:
            input_ids: 分词后的输入 (batch_size, num_attributes, max_len)
            attention_mask: 注意力掩码 (batch_size, num_attributes, max_len)
        
        返回:
            证据向量 (batch_size, num_attributes, hidden_size)
        """
        # 对所有属性对进行编码
        # 由于我们格式化为 "[CLS] left [SEP] right [SEP]"，编码器
        # 已经在 [CLS] 标记中生成了组合表示
        evidence_vectors = self.encode_attribute_pairs(input_ids, attention_mask)
        
        return evidence_vectors
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        通过完整模型的前向传播。
        """
        # 1. 构建证据
        evidence_vectors = self.construct_evidence_vectors(input_ids, attention_mask)
        
        # 2. 因果聚合 (CAN)
        aggregated_vector, gate_values = self.can(evidence_vectors)
        
        # --- [新增代码] 特征归一化 (L2 Normalize) ---
        # 这一步是为了对比学习，把向量投影到单位球面上
        features = F.normalize(aggregated_vector, p=2, dim=1)
        # ----------------------------------------
        
        # 3. 分类 (依然使用 aggregated_vector 进入分类器)
        logits = self.classifier(aggregated_vector)
        
        return {
            "logits": logits,
            "gates": gate_values,
            "evidence": evidence_vectors,
            "features": features  # <--- [必须添加] 返回归一化特征给 Trainer
        }
    
    def get_gate_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        获取每个属性的因果重要性（门控值）。
        用于模型解释和分析。
        
        Args:
            input_ids: 分词后的输入
            attention_mask: 注意力掩码
        
        Returns:
            门控值 (batch_size, num_attributes)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs["gates"]


def count_parameters(model: nn.Module) -> int:
    """
    统计模型中的可训练参数数量。
    
    Args:
        model: PyTorch 模型
    
    Returns:
        可训练参数的数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> CIMModel:
    """
    初始化和配置 CIM 模型的工厂函数。
    
    Args:
        model_config: 模型配置
        dataset_config: 数据集配置
        device: 放置模型的设备
    
    Returns:
        初始化的 CIMModel
    """
    model = CIMModel(model_config, dataset_config)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"已初始化 CIM 模型")
    print(f"属性数量: {len(dataset_config.attributes)}")
    
    return model
