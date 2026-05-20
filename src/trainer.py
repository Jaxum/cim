"""
因果实体解析(CIM)框架的训练器模块。
本模块作为"训练指挥官"，实现：
- 因果干预训练(CIT)损失计算
- 训练和评估循环
- 模型检查点保存和加载
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from typing import Dict, Optional

import wandb
import math


from .config import MainConfig, ModelConfig
from .model import CIMModel
from .dataset import CausalERDataset


class FocalLoss(nn.Module):
    """用于分类的 Focal Loss，以应对类别不平衡问题。

    参数:
        gamma: 聚焦参数，用于调节对困难样本的关注度
        alpha: 正类的加权系数（0~1 之间的浮点数）
        reduction: 损失的聚合方式（'none'、'mean'、'sum'）
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.9, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch, num_classes), targets: (batch,)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            # alpha是正类 (class 1) 的加权系数
            # 为每个样本构建权重
            alpha_factor = torch.ones_like(targets, dtype=logits.dtype, device=logits.device) * (1 - self.alpha)
            alpha_factor = torch.where(targets == 1, torch.ones_like(alpha_factor) * self.alpha, alpha_factor)
            focal_loss = alpha_factor * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """
    因果实体解析模型的训练器。
    
    实现CIM的第二支柱：因果干预训练(CIT)，
    通过反事实学习教会模型区分因果证据和混淆因素。
    """
    
    def __init__(
        self,
        model: CIMModel,
        main_config: MainConfig,
        model_config: ModelConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        use_wandb: bool = False
    ):
        """
        初始化训练器。
        
        参数:
            model: 要训练的CIM模型
            main_config: 主配置
            model_config: 模型配置
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            test_loader: 测试数据加载器(可选)
            use_wandb: 是否使用Weights & Biases记录日志
        """
        self.model = model
        self.main_config = main_config
        self.model_config = model_config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.use_wandb = use_wandb
        
        # 存储数据集配置以检查属性
        self.dataset_config = model.dataset_config
        
        # 设备
        self.device = next(model.parameters()).device
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=model_config.lr,
            weight_decay=model_config.weight_decay
        )
        
        # 学习率调度器
        # 注意：scheduler.step() 只在“真正进行 optimizer.step()”时调用一次。
        # 因此这里的 total_steps 应该按 optimizer step 计数，而不是 batch step。
        steps_per_epoch = int(math.ceil(len(train_loader) / max(1, model_config.gradient_accumulation_steps)))
        total_steps = steps_per_epoch * model_config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=model_config.warmup_steps,
            num_training_steps=total_steps
        )
        
        
        self.ce_loss_fn = FocalLoss(gamma=model_config.focal_gamma, alpha=model_config.focal_alpha)

        # --- [新增代码] 初始化对比损失函数 ---
        # 1. PUSH (因果): 推远向量。margin 读取自 config
        self.push_loss_fn = nn.CosineEmbeddingLoss(margin=model_config.causal_margin)

        # 混合精度训练
        self.use_fp16 = model_config.use_fp16 and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_fp16 else None  # type: ignore
        
        # 打印FP16状态
        if self.use_fp16:
            print(f"✓ FP16混合精度训练已启用")
        else:
            print(f"⚠ FP16未启用 (CUDA可用: {torch.cuda.is_available()}, use_fp16配置: {model_config.use_fp16})")
        
        # 最优阈值（通过阈值搜索得到）
        self.best_threshold = 0.5

        # 最佳模型跟踪
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        # 如果需要则初始化wandb
        if self.use_wandb:
            wandb.init(
                project="causal-er",
                name=f"{main_config.dataset_name}_{model_config.encoder_model_name}",
                config={
                    "dataset": main_config.dataset_name,
                    "model": model_config.encoder_model_name,
                    "lambda1": model_config.lambda1,
                    "lambda2": model_config.lambda2,
                    "lr": model_config.lr,
                    "batch_size": model_config.train_batch_size,
                    "epochs": model_config.epochs
                }
            )
        
        print(f"训练器已初始化")
        print(f"总训练步数: {total_steps}")
        print(f"Lambda1 (因果): {model_config.lambda1}, Lambda2 (混淆): {model_config.lambda2}")
        print(f"CIT 分支启用: {getattr(model_config, 'enable_cit', True)}")
        # 尝试初始化分类器偏置以反映训练集中正样本比例（缓解从预测全负的情况）
        try:
            # 获取训练集标签统计
            train_dataset = self.train_loader.dataset
            if isinstance(train_dataset, CausalERDataset) and hasattr(train_dataset, 'data'):
                labels = train_dataset.data[self.dataset_config.label_column].astype(int).values
                labels_array = np.asarray(labels)  # 显式转为 numpy array
                pos = (labels_array == 1).sum()
                neg = (labels_array == 0).sum()
                if pos > 0 and neg > 0:
                    pos_prob = pos / (pos + neg)
                    # 计算 logit 以设置偏置: b1 - b0 = log(p/(1-p)), 我们将 b0 = 0, b1 = logit
                    logit_pos = float(np.log(pos_prob / (1 - pos_prob)))
                    final_linear = self.model.classifier[-1] if isinstance(self.model.classifier[-1], nn.Linear) else None
                    if final_linear is not None and final_linear.bias is not None:
                        with torch.no_grad():
                            # bo = 0, b1 = logit_pos
                            final_linear.bias.data = torch.tensor([0.0, logit_pos], device=self.device)
                            print(f"初始化分类器偏置以反映训练正样本比: pos_prob={pos_prob:.4f}, logit={logit_pos:.4f}")
        except Exception:
            pass
        
        # 检查并警告单属性或无混淆属性的数据集
        num_attrs = len(self.dataset_config.attributes)
        num_causal = len(self.dataset_config.causal_attributes)
        num_conf = len(self.dataset_config.confounder_attributes)
        
        if num_attrs == 1:
            print(f"    检测到单属性数据集: {self.dataset_config.name}")
        
        if num_conf == 0:
            print(f"    数据集中无混淆属性: {self.dataset_config.name}")
            print(f"    训练期间将跳过混淆损失(λ₂)。")

        # 门控权重追踪提示
        if getattr(self.model_config, "gate_track_enabled", False):
            print("    门控权重追踪: 已启用")
            print(f"    追踪范围: {self.model_config.gate_track_scope}")
            print(f"    输出目录: {self.model_config.gate_track_out_dir}")

    def _get_dataset_short_name(self) -> str:
        """用于门控权重追踪输出的短数据集名。"""
        try:
            base = os.path.basename(str(self.dataset_config.data_path)).strip()
            return base if base else str(self.dataset_config.name)
        except Exception:
            return str(self.dataset_config.name)

    def _collect_gate_means(
        self,
        loader: DataLoader,
        max_batches: int = 0,
    ) -> Dict[str, float]:
        """统计一个数据集加载器上的门控权重均值（按属性）。"""

        self.model.eval()
        attr_names = list(self.dataset_config.attributes)
        num_attrs = len(attr_names)

        gate_sum = torch.zeros(num_attrs, device=self.device)
        gate_count = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches > 0 and i >= max_batches:
                    break
                input_ids = batch["normal_input_ids"].to(self.device)
                attention_mask = batch["normal_attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                if isinstance(outputs, dict):
                    gates = outputs.get("gates")
                elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                    gates = outputs[1]
                else:
                    gates = getattr(outputs, "gating_weights", None)

                if gates is None:
                    raise ValueError("模型输出中未找到 gates，无法统计门控权重")

                gate_sum += gates.sum(dim=0)
                gate_count += gates.size(0)

        if gate_count == 0:
            raise ValueError("门控权重统计失败：没有可用样本")

        gate_mean = (gate_sum / gate_count).detach().cpu().numpy().tolist()

        # 根据 scope 过滤输出
        scope = getattr(self.model_config, "gate_track_scope", "all")
        if scope == "causal_conf":
            keep = set(self.dataset_config.causal_attributes) | set(self.dataset_config.confounder_attributes)
            keep = {k for k in keep if k and str(k).strip()}
        else:
            keep = set(attr_names)

        return {a: m for a, m in zip(attr_names, gate_mean) if a in keep}

    def _save_gate_trace(self, epoch: int, gate_means: Dict[str, float]) -> None:
        """将当前 epoch 的门控权重均值追加到 CSV。"""

        out_root = getattr(self.model_config, "gate_track_out_dir", "./gate_traces")
        dataset_short = self._get_dataset_short_name()
        out_dir = os.path.join(out_root, dataset_short)
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "gates_over_epochs.csv")

        # 若文件不存在，先写表头
        if not os.path.exists(csv_path):
            header = ["epoch"] + list(gate_means.keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")

        # 追加当前 epoch 行
        values = [str(epoch)] + [f"{gate_means[k]:.6f}" for k in gate_means.keys()]
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(values) + "\n")

    
    def _compute_cit_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算损失函数：几何对比学习 + R-Drop 正则化
        Lose =
            loss_ce +                       分类loss 
            loss_rdrop +                    R-Drop正则化
            lambda1 * loss_causal +         因果干预损失 (Push)
            lambda2 * loss_confounder       混淆属性损失（pull）
        """
        
        # 1. 准备数据
        normal_input_ids = batch["normal_input_ids"].to(self.device)
        normal_attention_mask = batch["normal_attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        is_match = batch["is_match"].to(self.device)
        
        # R-Drop：当 rdrop_alpha<=0 时，跳过第二次前向（显著提速）
        use_rdrop = getattr(self.model_config, 'rdrop_alpha', 0.0) is not None and self.model_config.rdrop_alpha > 0

        if use_rdrop:
            # === [R-Drop 核心修改 A: 拼接输入] ===
            # 将输入复制一份，在 Batch 维度拼接
            # 形状变化: [Batch, Len] -> [2*Batch, Len]
            combined_input_ids = torch.cat([normal_input_ids, normal_input_ids], dim=0)
            combined_mask = torch.cat([normal_attention_mask, normal_attention_mask], dim=0)

            # 2. 前向传播 (一次性跑两份数据)
            combined_outputs = self.model(combined_input_ids, combined_mask)
            combined_logits = combined_outputs["logits"]
            combined_feats = combined_outputs.get("features")

            if combined_feats is None:
                raise ValueError("model.py 未返回 'features'，请检查 src/model.py")

            # === [R-Drop 核心修改 B: 拆分输出] ===
            batch_size = normal_input_ids.size(0)
            # 前半部分是第一次跑的结果，后半部分是第二次跑的结果
            logits_1 = combined_logits[:batch_size]
            logits_2 = combined_logits[batch_size:]

            # 特征只需要取第一份，用于后续的对比学习
            normal_feats = combined_feats[:batch_size]

            # === [R-Drop 核心修改 C: 计算损失] ===
            # 1. 主任务分类损失 (使用两份的平均值，取平均更稳)
            loss_ce = 0.5 * (self.ce_loss_fn(logits_1, labels) + self.ce_loss_fn(logits_2, labels))

            # 2. R-Drop KL 散度损失 (双向)
            p = F.log_softmax(logits_1, dim=-1)
            q = F.log_softmax(logits_2, dim=-1)
            p_tec = F.softmax(logits_1, dim=-1)
            q_tec = F.softmax(logits_2, dim=-1)

            kl_loss = F.kl_div(p, q_tec, reduction='batchmean') + F.kl_div(q, p_tec, reduction='batchmean')
            loss_rdrop = self.model_config.rdrop_alpha * 0.5 * kl_loss
        else:
            outputs = self.model(normal_input_ids, normal_attention_mask)
            logits = outputs["logits"]
            normal_feats = outputs.get("features")
            if normal_feats is None:
                raise ValueError("model.py 未返回 'features'，请检查 src/model.py")

            loss_ce = self.ce_loss_fn(logits, labels)
            loss_rdrop = torch.tensor(0.0, device=self.device)

        # 对比学习 (Push & Pull)
        # 注意：这里我们使用 normal_feats (即 feats_1) 作为 Anchor
        use_cit = getattr(self.model_config, 'enable_cit', True)
        
        loss_causal = torch.tensor(0.0, device=self.device)
        loss_confounder = torch.tensor(0.0, device=self.device)
        num_positive = is_match.sum().item()
        
        if use_cit and num_positive > 0:
            anchor_feats = normal_feats[is_match]
            pos_indices_cpu = batch["is_match"]
            
            # --- 因果干预 (Push) ---
            causal_input_ids = batch["causal_input_ids"][pos_indices_cpu].to(self.device)
            causal_mask = batch["causal_attention_mask"][pos_indices_cpu].to(self.device)
            
            causal_outputs = self.model(causal_input_ids, causal_mask)
            negative_feats = causal_outputs["features"]
            
            target_push = -torch.ones(int(num_positive), device=self.device)
            loss_causal = self.push_loss_fn(anchor_feats, negative_feats, target_push)
            
            # --- 混淆干预 (Pull) ---
            if self.dataset_config.confounder_attributes:
                conf_input_ids = batch["conf_input_ids"][pos_indices_cpu].to(self.device)
                conf_mask = batch["conf_attention_mask"][pos_indices_cpu].to(self.device)
                
                conf_outputs = self.model(conf_input_ids, conf_mask)
                positive_feats = conf_outputs["features"]
                
                cos_sim = F.cosine_similarity(anchor_feats, positive_feats, dim=1)
                threshold = getattr(self.model_config, 'confounder_margin', 0.9)
                loss_confounder = F.relu(threshold - cos_sim).mean()
        
        # === 总损失聚合 ===
        # Total = CE + R-Drop + lambda1*Push + lambda2*Pull
        total_loss = (
            loss_ce +
            loss_rdrop +
            self.model_config.lambda1 * loss_causal +
            self.model_config.lambda2 * loss_confounder
        )
        
        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_rdrop": loss_rdrop, # 方便看日志
            "loss_causal": loss_causal,
            "loss_confounder": loss_confounder,
            "num_positive": torch.tensor(num_positive)
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch。
        
        核心流程：
        1. 遍历所有训练batch
        2. 对每个batch进行前向传播和反向传播
        3. 累积指定数量的梯度后，才更新一次模型参数
        4. 支持FP16混合精度加速训练
        
        参数:
            epoch: 当前epoch编号
        
        返回:
            包含该epoch平均损失的字典
        """
        self.model.train()
        
        # 初始化损失累积器
        total_loss = 0.0
        total_ce_loss = 0.0
        total_causal_loss = 0.0
        total_conf_loss = 0.0
        num_batches = 0
        
        # 创建进度条
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 前向传播（计算损失） 
            # 使用FP16混合精度（如果启用）加速前向传播
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):  # type: ignore
                # 计算CIT损失（包含正常损失、因果损失、混淆损失）
                loss_dict = self._compute_cit_loss(batch)
                loss = loss_dict["loss"]
            
            # 反向传播（计算梯度）
            if self.scaler is not None:
                # FP16模式：缩放损失以防止梯度下溢
                self.scaler.scale(loss).backward()
            else:
                # FP32模式：直接反向传播
                loss.backward()
            
            # 梯度累积 + 参数更新
            # 关键：只有累积够gradient_accumulation_steps次梯度，才真正更新参数
            # 效果：等价于batch_size扩大gradient_accumulation_steps倍
            if (step + 1) % self.model_config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # FP16模式：
                    # 1. 先反缩放梯度到原始尺度（只在更新前调用一次）
                    self.scaler.unscale_(self.optimizer)
                    # 2. 裁剪梯度范数到max_grad_norm（例如1.0）
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.model_config.max_grad_norm
                    )
                    # 3. 更新模型参数（使用累积的梯度）
                    self.scaler.step(self.optimizer)
                    # 4. 更新FP16的缩放因子（动态调整以避免溢出）
                    self.scaler.update()
                    # 5. 清空梯度缓存，为下一轮累积做准备
                    self.optimizer.zero_grad()
                    # 6. 更新学习率（warmup + 线性衰减）
                    self.scheduler.step()
                else:
                    # FP32模式的相同逻辑
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.model_config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # 记录损失和进度
            # 累积当前batch的损失（用于计算epoch平均值）
            total_loss += loss.item()
            total_ce_loss += loss_dict["loss_ce"].item()
            total_causal_loss += loss_dict["loss_causal"].item()
            total_conf_loss += loss_dict["loss_confounder"].item()
            num_batches += 1
            
            # 更新进度条显示（实时显示当前batch的损失）
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{loss_dict['loss_ce'].item():.4f}",
                "causal": f"{loss_dict['loss_causal'].item():.4f}",
                "conf": f"{loss_dict['loss_confounder'].item():.4f}"
            })
            
            # 记录到wandb（如果启用了日志）
            if self.use_wandb and step:
                wandb.log({  # type: ignore
                    "train/loss": loss.item(),
                    "train/ce_loss": loss_dict["loss_ce"].item(),
                    "train/causal_loss": loss_dict["loss_causal"].item(),
                    "train/conf_loss": loss_dict["loss_confounder"].item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": step
                })
        
        # 计算epoch的平均损失
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_causal_loss = total_causal_loss / num_batches
        avg_conf_loss = total_conf_loss / num_batches
        
        return {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
            "causal_loss": avg_causal_loss,
            "conf_loss": avg_conf_loss
        }
    
    def evaluate(self, data_loader: DataLoader, split: str = "valid", threshold: float = 0.5) -> Dict[str, float]:
        """
        在数据集上评估模型。
        
        参数:
            data_loader: 评估用的数据加载器
            split: 数据集划分名称("valid"或"test")
            threshold: 分类阈值，默认0.5
        
        返回:
            包含评估指标的字典
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []  # 收集概率用于调试
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                # 移到设备
                input_ids = batch["normal_input_ids"].to(self.device)
                attention_mask = batch["normal_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # 使用混合精度加速评估
                with torch.amp.autocast('cuda', enabled=self.scaler is not None): # type: ignore
                    # 前向传播
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["logits"]
                    
                    # 计算损失
                    loss = self.ce_loss_fn(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 获取预测和概率
                probs = F.softmax(logits, dim=-1)
                # 使用阈值进行预测（而不是argmax）
                predictions = (probs[:, 1] >= threshold).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        avg_loss = total_loss / num_batches
        
        # 调试输出：混淆矩阵与预测统计
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_predictions)
        pred_pos = int((all_predictions == 1).sum())
        true_pos = int((all_labels == 1).sum())
        
        print(f"  混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        print(f"  预测正样本: {pred_pos}/{len(all_predictions)} ({pred_pos/len(all_predictions)*100:.2f}%)")
        print(f"  真实正样本: {true_pos}/{len(all_labels)} ({true_pos/len(all_labels)*100:.2f}%)")
        
        metrics = {
            f"{split}/loss": avg_loss,
            f"{split}/accuracy": accuracy,
            f"{split}/precision": precision,
            f"{split}/recall": recall,
            f"{split}/f1": f1
        }
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log(metrics)  # type: ignore
        
        return metrics
    
    def train(self):
        """
        主训练循环。
        """
        print("\n" + "="*80)
        print("开始因果干预训练 (CIT)")
        print("="*80 + "\n")

        # 记录 epoch=0 的门控权重（训练前）
        if getattr(self.model_config, "gate_track_enabled", False):
            try:
                max_batches = int(getattr(self.model_config, "gate_track_max_batches", 0) or 0)
                gate_means = self._collect_gate_means(self.valid_loader, max_batches=max_batches)
                self._save_gate_trace(0, gate_means)
                print("✓ 已记录 epoch=0 门控权重")
            except Exception as e:
                print(f"⚠ epoch=0 门控权重追踪失败：{e}")
        
        for epoch in range(1, self.model_config.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.model_config.epochs} ---")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, "
                  f"Causal: {train_metrics['causal_loss']:.4f}, "
                  f"Conf: {train_metrics['conf_loss']:.4f})")
            
            # 在验证集上评估
            valid_metrics = self.evaluate(self.valid_loader, split="valid")
            print(f"Valid F1: {valid_metrics['valid/f1']:.4f}, "
                  f"Precision: {valid_metrics['valid/precision']:.4f}, "
                  f"Recall: {valid_metrics['valid/recall']:.4f}")

            # 记录门控权重（不影响训练流程，可选）
            if getattr(self.model_config, "gate_track_enabled", False):
                try:
                    max_batches = int(getattr(self.model_config, "gate_track_max_batches", 0) or 0)
                    gate_means = self._collect_gate_means(self.valid_loader, max_batches=max_batches)
                    self._save_gate_trace(epoch, gate_means)
                    print("✓ 已记录本轮门控权重")
                except Exception as e:
                    print(f"⚠ 门控权重追踪失败：{e}")
            
            # 保存最佳模型(不重新加载，继续用当前模型训练)
            if valid_metrics['valid/f1'] >= self.best_f1:
                self.best_f1 = valid_metrics['valid/f1']
                self.best_epoch = epoch
                self.save_model("best_model.pt")
                print(f"✓ 保存新的最佳模型! F1: {self.best_f1:.4f}")
            
        
        print("\n" + "="*80)
        print(f"训练完成! 最佳 F1: {self.best_f1:.4f}，发生在第 {self.best_epoch} 轮")
        print("="*80 + "\n")

        # 阈值搜索已迁移到 matcher.py（推理/评估阶段执行）。

        
    
    def search_optimal_threshold(self) -> tuple[float, float]:
        """
        在验证集上搜索最优分类阈值。
        
        返回:
            (最优阈值, 对应的最佳F1分数)
        """
        self.model.eval()
        
        # 收集验证集上的所有预测概率和标签
        all_probs = []
        all_labels = []
        
        print("收集验证集预测概率...")
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="收集概率"):
                input_ids = batch["normal_input_ids"].to(self.device)
                attention_mask = batch["normal_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                with torch.amp.autocast('cuda', enabled=self.use_fp16):  # type: ignore
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs["logits"]
                    probs = F.softmax(logits, dim=-1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 搜索最优阈值
        best_threshold = 0.5
        best_f1 = 0.0
        
        thresholds = np.arange(
            self.model_config.threshold_search_start,
            self.model_config.threshold_search_end + self.model_config.threshold_search_step,
            self.model_config.threshold_search_step
        )
        
        print(f"\n搜索阈值范围: [{self.model_config.threshold_search_start}, {self.model_config.threshold_search_end}], 步长: {self.model_config.threshold_search_step}")
        print(f"{'阈值':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} ")
        print("-" * 60)
        
        for threshold in thresholds:
            predictions = (all_probs >= threshold).astype(int)
            
            f1 = f1_score(all_labels, predictions, zero_division=0)
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            accuracy = accuracy_score(all_labels, predictions)
            
            print(f"{threshold:<10.3f} {f1:<10.4f} {precision:<12.4f} {recall:<10.4f} ")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print("-" * 60)
        print(f"最优阈值: {best_threshold:.3f}, 最佳F1: {best_f1:.4f}")
        
        return float(best_threshold), float(best_f1)
    
    def save_model(self, filename: str):
        """
        保存模型检查点。
        
        参数:
            filename: 检查点文件名
        """
        os.makedirs(self.main_config.output_dir, exist_ok=True)
        filepath = os.path.join(self.main_config.output_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "model_config": self.model_config,
        }
        
        torch.save(checkpoint, filepath)
        print(f"模型已保存至: {filepath}")
    
    def load_model(self, filename: str):
        """
        加载模型检查点。
        
        参数:
            filename: 检查点文件名
        """
        filepath = os.path.join(self.main_config.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_f1 = checkpoint.get("best_f1", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_threshold = checkpoint.get("best_threshold", 0.5)
        
        print(f"模型已从以下路径加载: {filepath}")
        print(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}, Threshold: {self.best_threshold:.3f}")
