"""Hard Subset（困难子集）评测工具。

本模块用于构造并评估“高重叠困难负例（hard negatives）”相关的子集，以衡量模型
对捷径学习（shortcut learning）的鲁棒性。

定义：hard negative 是指满足以下条件的负样本对：
1) 标签为 0（非匹配）；
2) 左右实体文本的词汇重叠度很高。

我们使用 token-level 的 Jaccard 相似度来度量左右实体的重叠度：
J(A,B)=|A∩B|/|A∪B|。


"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    """将文本切分为 token 集合（小写、去重）。

    这里使用一个非常简单的正则抽取字母/数字串，目的是得到稳定、可复现的
    lexical overlap 指标，而不是追求最强的分词质量。
    """
    if not text:
        return set()
    return set(t.lower() for t in _TOKEN_RE.findall(text))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """计算两个 token 集合的 Jaccard 相似度。"""
    if not a and not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    return float(len(inter) / max(1, len(union)))


def _concat_side_values(row: "dict", attributes: Sequence[str], side_prefix: str) -> str:
    """将某一侧（left/right）的多个属性值拼接成一个文本串。

    Args:
        row: 单条样本的字典（通常来自 DataFrame 的一行）。
        attributes: 要参与拼接的属性列表。
        side_prefix: 'ltable_' 或 'rtable_'，用于定位列名。
    """
    parts: List[str] = []
    for attr in attributes:
        key = f"{side_prefix}{attr}"
        val = row.get(key, "")
        if val is None:
            val = ""
        val_str = str(val).strip()
        if val_str:
            parts.append(val_str)
    return " ".join(parts)


def compute_pair_jaccard(row: "dict", attributes: Sequence[str]) -> float:
    """对一个样本对（row）计算左右实体文本的 Jaccard 重叠度。"""
    left_text = _concat_side_values(row, attributes, "ltable_")
    right_text = _concat_side_values(row, attributes, "rtable_")
    return jaccard(_tokenize(left_text), _tokenize(right_text))


def get_hard_negative_indices(
    df,  # pandas.DataFrame：为避免强依赖，这里不写死类型注解
    attributes: Sequence[str],
    label_column: str = "label",
    jaccard_threshold: float = 0.6,
    candidate_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """返回 hard negatives 的数据行索引列表。

    hard negative 定义：label==0 且 Jaccard(left,right) >= 阈值。

    Args:
        df: DataFrame，需包含 ltable_<attr> / rtable_<attr> 以及 label 列。
        attributes: 用于拼接计算重叠度的属性列表。
        label_column: 标签列名。
        jaccard_threshold: hard negative 的重叠度阈值。
        candidate_indices: 可选，仅在该索引子集上筛选（用于加速或子集评估）。

    Returns:
        df 的行索引列表（int）。
    """
    if candidate_indices is None:
        indices_iter: Iterable[int] = range(len(df))
    else:
        indices_iter = candidate_indices

    hard: List[int] = []

    # 优化：尽量一次性读取 label，避免循环中频繁访问 DataFrame
    try:
        labels = df[label_column].astype(int).to_numpy()
    except Exception:
        labels = None

    for i in indices_iter:
        if labels is not None:
            if int(labels[i]) != 0:
                continue
            row = df.iloc[int(i)].to_dict()
        else:
            row = df.iloc[int(i)].to_dict()
            if int(row.get(label_column, 0)) != 0:
                continue

        score = compute_pair_jaccard(row, attributes)
        if score >= jaccard_threshold:
            hard.append(int(i))

    return hard


def get_hard_subset_indices(
    df,
    attributes: Sequence[str],
    label_column: str = "label",
    jaccard_threshold: float = 0.6,
    candidate_indices: Optional[Sequence[int]] = None,
) -> dict:
    """返回 Hard Subset 评测所需的索引集合。

    我们采用 anti-shortcut 的常用评测协议：
    - hard negatives：label==0 且 Jaccard(left,right) >= 阈值
    - hard subset（用于计算 F1）：所有正例（label==1）∪ hard negatives

    这样做的原因：如果只在负例子集上评估，F1/Recall 在数学上会失去意义。
    使用“正例 + 高重叠负例”的组合子集，能够同时反映：
    1) 模型是否避免在高重叠负例上误报（降低 shortcut 诱导的 FP）；
    2) 模型是否仍能保住正例召回（TP 不被过度压制）。

    Returns:
        包含 hard negatives 与 hard subset 的索引，以及正负样本统计。
    """
    if candidate_indices is None:
        candidate_indices = list(range(len(df)))

    # labels
    labels = df[label_column].astype(int).to_numpy()
    cand = np.asarray(candidate_indices, dtype=int)

    pos_mask = labels[cand] == 1
    neg_mask = labels[cand] == 0

    pos_indices = cand[pos_mask].tolist()
    neg_indices = cand[neg_mask].tolist()

    hard_neg_indices = get_hard_negative_indices(
        df,
        attributes=attributes,
        label_column=label_column,
        jaccard_threshold=jaccard_threshold,
        candidate_indices=neg_indices,
    )

    hard_subset_indices = sorted(set(pos_indices).union(hard_neg_indices))

    return {
        "hard_neg_indices": hard_neg_indices,
        "hard_subset_indices": hard_subset_indices,
        "num_pos": int(len(pos_indices)),
        "num_neg": int(len(neg_indices)),
    }


def summarize_hard_subset(
    df,
    attributes: Sequence[str],
    label_column: str = "label",
    jaccard_threshold: float = 0.6,
) -> dict:
    """统计 hard negatives 的规模（用于日志/论文报告）。"""
    labels = df[label_column].astype(int).to_numpy()
    neg_indices = np.where(labels == 0)[0]
    hard_indices = get_hard_negative_indices(
        df,
        attributes=attributes,
        label_column=label_column,
        jaccard_threshold=jaccard_threshold,
        candidate_indices=neg_indices.tolist(),
    )
    return {
        "num_total": int(len(df)),
        "num_neg": int(len(neg_indices)),
        "num_hard_neg": int(len(hard_indices)),
        "hard_neg_ratio_in_neg": float(len(hard_indices) / max(1, len(neg_indices))),
    }
