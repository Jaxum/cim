"""轻量级 LLM 属性自动标注脚本（causal/confounder/uncertain）。

Lightweight LLM-assisted attribute labeling for causal/confounder/uncertain.

This script reads the first N record pairs from a dataset split, sends schema +
examples to an LLM, and asks it to label each attribute using three rules:
1) Counterfactual mutability
2) Environmental invariance
3) Semantic exclusivity

Output is a JSON report with per-vote and aggregated labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from urllib import error, request

from src.config import DATASET_METADATA, get_dataset_config


OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"


def sanitize_name(name: str) -> str:
    # 用于路径命名，避免模型名中的斜杠、冒号等非法字符。
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return safe.strip("._-") or "unknown"


def log_step(message: str) -> None:
    # 统一的运行过程输出，便于定位卡在哪一步。
    print(f"[llm_label] {message}", flush=True)


def build_label_cache_path(provider: str, model: str, dataset_name: str) -> str:
    # 每个 LLM + 数据集一个稳定文件。
    out_dir = os.path.join(
        "./attr_auto",
        "llm_labels_cache",
        sanitize_name(provider),
        sanitize_name(model),
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{sanitize_name(dataset_name)}_selected_labels.json")


def load_cached_selection(cache_path: str) -> Dict[str, object]:
    with open(cache_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid cache format: {cache_path}")
    if "selected" not in obj or not isinstance(obj["selected"], dict):
        raise ValueError(f"Invalid cache format (missing selected): {cache_path}")
    return obj


def build_selected_labels(final: Dict[str, Dict[str, object]]) -> Dict[str, List[str]]:
    selected = {
        "causal": [],
        "confounder": [],
        "uncertain": [],
    }
    for attr, item in final.items():
        label = str(item.get("final_label", "uncertain"))
        if label not in selected:
            label = "uncertain"
        selected[label].append(attr)
    return selected


def save_selected_labels(
    cache_path: str,
    provider: str,
    model: str,
    dataset_name: str,
    split: str,
    selected: Dict[str, List[str]],
    final_by_attribute: Dict[str, Dict[str, object]],
) -> None:
    payload = {
        "provider": provider,
        "model": model,
        "dataset": dataset_name,
        "split": split,
        "selected": selected,
        "final_by_attribute": final_by_attribute,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-assisted attribute labeling (causal/confounder/uncertain)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="选择调用后端：openai 或 gemini",
    )
    parser.add_argument("--dataset_name", required=True, help="e.g. Amazon-Google")
    parser.add_argument("--data_root", default="./data", help="Dataset root directory")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--num_pairs", type=int, default=8, help="Use first N pairs as examples")
    parser.add_argument("--votes", type=int, default=3, help="Number of repeated LLM votes")
    parser.add_argument("--model", default=None, help="Override model name; default depends on provider")
    parser.add_argument("--api_base", default=None, help="Override API base URL; default depends on provider")
    parser.add_argument(
        "--api_key_env",
        default=None,
        help="Override API key env var; default depends on provider",
    )
    parser.add_argument(
        "--openai_model",
        default="gpt-4.1-mini",
        help="OpenAI backend default model name",
    )
    parser.add_argument(
        "--openai_api_base",
        default=OPENAI_DEFAULT_BASE,
        help="OpenAI backend base URL",
    )
    parser.add_argument(
        "--openai_api_key_env",
        default="OPENAI_API_KEY",
        help="Env var name for OpenAI API key",
    )
    parser.add_argument(
        "--gemini_model",
        default="gemini-2.0-flash",
        help="Gemini backend default model name",
    )
    parser.add_argument(
        "--gemini_api_base",
        default=GEMINI_DEFAULT_BASE,
        help="Gemini backend base URL",
    )
    parser.add_argument(
        "--gemini_api_key_env",
        default="GEMINI_API_KEY",
        help="Env var name for Gemini API key",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--request_timeout", type=float, default=240.0, help="HTTP request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=1.5)
    parser.add_argument(
        "--reuse_cached_labels",
        action="store_true",
        help="If cache exists for provider+model+dataset, skip API call and reuse it.",
    )
    parser.add_argument(
        "--refresh_cached_labels",
        action="store_true",
        help="Force refresh cache by calling API even if cached file exists.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print prompt preview and parsed samples, do not call API.",
    )
    return parser.parse_args()


def resolve_backend_config(args: argparse.Namespace) -> Dict[str, str]:
    """根据 provider 选择默认模型、接口地址和 key 环境变量。"""
    if args.provider == "openai":
        return {
            "model": args.model or args.openai_model,
            "api_base": args.api_base or args.openai_api_base,
            "api_key_env": args.api_key_env or args.openai_api_key_env,
        }
    if args.provider == "gemini":
        return {
            "model": args.model or args.gemini_model,
            "api_base": args.api_base or args.gemini_api_base,
            "api_key_env": args.api_key_env or args.gemini_api_key_env,
        }
    raise ValueError(f"Unsupported provider: {args.provider}")


def resolve_dataset_config_any(name: str):
    """Resolve dataset config by key, name, or data_path tail (alias)."""
    try:
        return get_dataset_config(name)
    except ValueError:
        pass

    for cfg in DATASET_METADATA.values():
        if name == cfg.name:
            return cfg

    norm = lambda s: s.replace("_", "").replace("-", "").lower()
    wanted = norm(name)
    for cfg in DATASET_METADATA.values():
        tail = os.path.basename(cfg.data_path)
        if norm(tail) == wanted:
            return cfg

    available = ", ".join(DATASET_METADATA.keys())
    raise ValueError(f"Dataset '{name}' not found. Available: {available}")


def parse_entity_text(entity_text: str, target_attrs: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}

    # Robust split parser for patterns like:
    # "COL <attr> VAL <value> COL <attr2> VAL <value2>"
    # It handles empty values (e.g., "VAL  COL next...") safely.
    # 数据行格式是 "COL <attr> VAL <value>" 的串联结构。
    parts = entity_text.split("COL ")
    for part in parts[1:]:
        if " VAL " not in part:
            continue
        attr, val = part.split(" VAL ", 1)
        key = attr.strip()
        if key in target_attrs:
            parsed[key] = " ".join(val.strip().split())

    for attr in target_attrs:
        parsed.setdefault(attr, "")
    return parsed


def read_pair_lines(file_path: str, num_pairs: int) -> List[Tuple[str, str, int]]:
    # 只读取前 N 对 pair，控制提示词长度与 API 成本。
    out: List[Tuple[str, str, int]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue

            parts = raw.split("\t")
            if len(parts) < 3:
                continue

            left = parts[0].strip()
            right = parts[1].strip()
            label_text = parts[-1].strip()
            try:
                label = int(label_text)
            except ValueError:
                label = -1

            out.append((left, right, label))
            if len(out) >= num_pairs:
                break
    return out


def infer_attr_type(values: List[str]) -> str:
    # 仅用于给 LLM 提供弱类型提示，不参与最终标签决策。
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "unknown"

    numeric_count = 0
    for v in non_empty:
        s = v.replace(",", "").replace("%", "").replace("$", "").strip()
        try:
            float(s)
            numeric_count += 1
        except ValueError:
            pass

    if numeric_count / len(non_empty) >= 0.7:
        return "numeric-like"
    return "text-like"


def build_schema_summary(samples: List[Dict[str, object]], attrs: List[str]) -> List[Dict[str, object]]:
    # 为每个属性提炼简要 schema：名称、类型提示、少量示例值。
    summary = []
    for attr in attrs:
        vals: List[str] = []
        for s in samples:
            left_obj = s.get("left", {})
            right_obj = s.get("right", {})
            left = left_obj if isinstance(left_obj, dict) else {}
            right = right_obj if isinstance(right_obj, dict) else {}
            vals.append(str(left.get(attr, "")))
            vals.append(str(right.get(attr, "")))

        uniq = []
        seen = set()
        for v in vals:
            vv = v.strip()
            if vv and vv not in seen:
                seen.add(vv)
                uniq.append(vv)
            if len(uniq) >= 4:
                break

        summary.append(
            {
                "name": attr,
                "type_hint": infer_attr_type(vals),
                "example_values": uniq,
            }
        )
    return summary


def build_prompt(dataset_name: str, attrs: List[str], schema_summary: List[Dict[str, object]], samples: List[Dict[str, object]]) -> str:
    # 提示词要求模型基于三条规则输出严格 JSON，便于后处理解析。
    payload = {
        "dataset": dataset_name,
        "target_attributes": attrs,
        "schema_summary": schema_summary,
        "pair_examples": samples,
    }

    instruction = (
    "You are an expert in Data Governance, Entity Matching and Causal Inference.\n\n"
    
    "Task: Classify each attribute into exactly one of three categories: [causal, confounder, uncertain].\n\n"
    
    "Apply the following three theoretical criteria jointly, with **Counterfactual Mutability as the primary criterion** when there is conflict:\n\n"
    
    "1. Counterfactual Mutability (Causal Criterion):\n"
    "Grounded in Pearl’s do-calculus, this criterion posits that if a substantive intervention (do(a_k = v')) on a specific attribute systematically alters the final entity matching label (i.e., changes whether two records refer to the same real-world entity), then a definitive causal relationship exists. "
    "Formally, an attribute belongs to the causal set if intervening on its value inherently mutates the ontology or physical composition of the entity. "
    "For example, modifying a product's exact model code, core technical specification, or a specific item's proper name fundamentally changes the entity’s identity, necessitating a label transition from Match to Non-match.\n\n"
    
    "2. Environmental Invariance (Spurious Criterion):\n"
    "Guided by Invariant Risk Minimization (IRM) theory, valid causal attributes remain stable across diverse environments, whereas spurious correlations fluctuate. "
    "Consequently, attributes exhibiting high spatio-temporal volatility or cross-platform variance such as dynamic sales prices, promotional tags,or user ratings are classified as confounders, since they do not alter the underlying physical entity itself.\n\n"
    
    "3. Semantic Exclusivity (Ontological Criterion):\n"
    "Drawing upon foundational record linkage theory, an attribute is considered causal if it possesses high discriminatory power and serves as a primary ontological identifier within the schema. "
    "Conversely, attributes providing broad, non-exclusive contextual background — such as general brand names, high-level taxonomic categories, macro-styles, or manufacturer/publisher names — lack unique identification power. These features often induce high-frequency false positives and are thus categorized as spurious confounders.\n\n"
    
    "Special Handling for 'uncertain':\n"
    "If the attribute is unstructured lengthy free-text (e.g., full 'description', 'review_text', or long summary), strictly classify it as 'uncertain' to avoid introducing mechanistic noise. Short proper names or primary titles should NOT fall into this category.\n\n"
    
    "Reasoning Requirement:\n"
    "For each attribute, provide concise reasoning that explicitly references how the three criteria apply, then derive the final label.\n\n"
    
    "Return STRICT JSON ONLY with this exact schema:\n"
    "{\n"
    "  \"attributes\":[\n"
    "    {\n"
    "      \"name\": \"<attribute_name>\",\n"
    "      \"reasoning\": \"<brief reasoning referencing the 3 criteria>\",\n"
    "      \"label\": \"causal|confounder|uncertain\",\n"
    "      \"confidence\": <float 0.0-1.0>\n"
    "    }\n"
    "  ]\n"
    "}\n"
    )

    return instruction + "\nInput:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json_obj(text: str) -> Dict[str, object]:
    # 兼容模型可能返回 markdown 代码块或额外文本的情况。
    text = text.strip()

    # 优先提取 markdown 代码块中的 json 内容。
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    def _try_parse(candidate: str) -> Dict[str, object] | None:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"attributes": obj}
        return None

    parsed = _try_parse(text)
    if parsed is not None:
        return parsed

    # 回退：从混合文本中提取第一个平衡的 JSON 对象。
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    parsed = _try_parse(candidate)
                    if parsed is not None:
                        return parsed
                    break

    preview = text.replace("\n", " ")[:220]
    raise ValueError(f"LLM response does not contain a JSON object. Preview: {preview}")


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    # 使用标准 Chat Completions 接口；失败时做有限重试。
    url = api_base.rstrip("/") + "/chat/completions"
    # Moonshot/KIMI 的部分模型仅接受 temperature=1。
    is_moonshot = "moonshot" in api_base.lower() or model.lower().startswith("moonshot")
    effective_temperature = 1.0 if is_moonshot else temperature
    body = {
        "model": model,
        "temperature": effective_temperature,
        "messages": [
            {"role": "system", "content": "You are a precise data annotation assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(body).encode("utf-8")

    last_err = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=request_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
        except error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="ignore")
            if e.code == 400 and "only 1 is allowed for this model" in err_text:
                raise RuntimeError(
                    "KIMI/Moonshot 模型要求 temperature=1。"
                    f"\n原始错误: HTTP {e.code}: {err_text}"
                ) from e
            last_err = f"HTTP {e.code}: {err_text}"
        except socket.timeout:
            last_err = (
                f"socket timeout after {request_timeout}s. "
                "可尝试提高 --request_timeout，或减少 --num_pairs / --votes。"
            )
        except Exception as e:  # noqa: BLE001
            last_err = str(e)

        if attempt < max_retries:
            time.sleep(retry_sleep)

    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def call_gemini_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    """调用 Gemini 原生 REST 接口。"""
    url = api_base.rstrip("/") + f"/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }
    data = json.dumps(body).encode("utf-8")

    last_err = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=request_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                candidates = payload.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"Gemini response has no candidates: {payload}")
                content = candidates[0].get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text_parts = []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                if not text_parts:
                    raise RuntimeError(f"Gemini response has no text parts: {payload}")
                return "".join(text_parts)
        except error.HTTPError as e:
            err_text = e.read().decode('utf-8', errors='ignore')
            last_err = f"HTTP {e.code}: {err_text}"

            # Gemini 免费额度耗尽时，直接给出明确提示，避免无意义重试。
            if e.code == 429:
                raise RuntimeError(
                    "Gemini 429 quota exceeded. 当前 API key 没有可用额度或 free tier 请求额度为 0。"
                    "\n可选处理方式："
                    "\n1) 切换到 OpenAI：--provider openai 并设置 OPENAI_API_KEY"
                    "\n2) 使用有付费额度的 Gemini 项目/Key"
                    "\n3) 更换 gemini 模型或项目后再试"
                    f"\n原始错误: {last_err}"
                ) from e
        except socket.timeout:
            last_err = (
                f"socket timeout after {request_timeout}s. "
                "可尝试提高 --request_timeout，或减少 --num_pairs / --votes。"
            )
        except Exception as e:  # noqa: BLE001
            last_err = str(e)

        if attempt < max_retries:
            time.sleep(retry_sleep)

    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def call_llm_completion(
    provider: str,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    """按 provider 路由到 OpenAI 或 Gemini 接口。"""
    if provider == "openai":
        return call_chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
    if provider == "gemini":
        return call_gemini_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def normalize_vote_result(raw_obj: Dict[str, object], attrs: List[str]) -> Dict[str, Dict[str, object]]:
    # 将单次投票结果规范化：标签集合受限，置信度裁剪到 [0,1]。
    attr_items = raw_obj.get("attributes", [])
    if not isinstance(attr_items, list):
        raise ValueError("Invalid LLM JSON: 'attributes' must be a list")

    by_name: Dict[str, Dict[str, object]] = {}
    for item in attr_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name in attrs:
            label = str(item.get("label", "uncertain")).strip().lower()
            if label not in {"causal", "confounder", "uncertain"}:
                label = "uncertain"
            conf = item.get("confidence", 0.0)
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = 0.0
            conf_f = max(0.0, min(1.0, conf_f))

            by_name[name] = {
                "label": label,
                "confidence": conf_f,
                "reasoning": str(item.get("reasoning", "")).strip(),
            }

    for attr in attrs:
        by_name.setdefault(
            attr,
            {
                "label": "uncertain",
                "confidence": 0.0,
                "reasoning": "",
            },
        )
    return by_name


def aggregate_votes(votes: List[Dict[str, Dict[str, object]]], attrs: List[str]) -> Dict[str, Dict[str, object]]:
    # 多次投票聚合：多数票决定最终标签，平票则回退 uncertain。
    result: Dict[str, Dict[str, object]] = {}
    for attr in attrs:
        labels = [str(v[attr].get("label", "uncertain")) for v in votes]
        confs: List[float] = []
        for v in votes:
            try:
                confs.append(float(str(v[attr].get("confidence", 0.0))))
            except (TypeError, ValueError):
                confs.append(0.0)
        counts = Counter(labels)

        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            final_label = "uncertain"
        else:
            final_label = top[0][0]

        result[attr] = {
            "final_label": final_label,
            "vote_labels": labels,
            "vote_counts": dict(counts),
            "mean_confidence": float(statistics.mean(confs)) if confs else 0.0,
            "per_vote": [v[attr] for v in votes],
        }
    return result


def build_output_path(dataset_name: str, split: str, output_file: str | None) -> str:
    if output_file:
        return output_file
    out_dir = os.path.join("./attr_auto", "llm_labels")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{dataset_name}_{split}_llm_labels.json")


def main() -> None:
    # 主流程：加载数据 -> 构造样本与提示词 -> LLM 多次投票 -> 聚合输出。
    args = parse_args()
    log_step(f"start provider={args.provider}, dataset={args.dataset_name}, split={args.split}")
    cfg = resolve_dataset_config_any(args.dataset_name)
    backend = resolve_backend_config(args)
    log_step(f"backend model={backend['model']}, api_base={backend['api_base']}")

    split_file = os.path.join(args.data_root, cfg.data_path, f"{args.split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    log_step(f"reading examples from {split_file} (num_pairs={args.num_pairs})")
    raw_pairs = read_pair_lines(split_file, args.num_pairs)
    if not raw_pairs:
        raise ValueError(f"No valid pairs found in {split_file}")
    log_step(f"loaded {len(raw_pairs)} pairs")

    pair_examples: List[Dict[str, object]] = []
    for left_text, right_text, label in raw_pairs:
        pair_examples.append(
            {
                "label": label,
                "left": parse_entity_text(left_text, cfg.attributes),
                "right": parse_entity_text(right_text, cfg.attributes),
            }
        )

    log_step("building schema summary and prompt")
    schema_summary = build_schema_summary(pair_examples, cfg.attributes)
    prompt = build_prompt(args.dataset_name, cfg.attributes, schema_summary, pair_examples)
    cache_path = build_label_cache_path(args.provider, backend["model"], args.dataset_name)
    log_step(f"cache path: {cache_path}")

    # dry_run 用于检查样本解析和提示词，不调用外部 API。
    if args.dry_run:
        log_step("dry_run enabled, skip API call")
        preview = {
            "dataset": args.dataset_name,
            "provider": args.provider,
            "model": backend["model"],
            "api_base": backend["api_base"],
            "cache_path": cache_path,
            "split_file": split_file,
            "num_pairs": len(pair_examples),
            "attributes": cfg.attributes,
            "schema_summary": schema_summary,
            "first_pair": pair_examples[0],
            "prompt_preview": prompt[:3000],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    # 可选：直接复用已有的每 LLM+数据集选择结果，不重复调用 API。
    if args.reuse_cached_labels and (not args.refresh_cached_labels) and os.path.exists(cache_path):
        log_step("reuse_cached_labels enabled and cache exists, loading cache")
        cached = load_cached_selection(cache_path)
        selected = cached.get("selected", {})
        report = {
            "dataset": args.dataset_name,
            "split": args.split,
            "source_file": split_file,
            "num_pairs_used": len(pair_examples),
            "votes": 0,
            "provider": args.provider,
            "model": backend["model"],
            "api_base": backend["api_base"],
            "attributes": cfg.attributes,
            "schema_summary": schema_summary,
            "final_by_attribute": cached.get("final_by_attribute", {}),
            "grouped_labels": selected,
            "selected_labels": selected,
            "pair_examples": pair_examples,
            "used_cached_labels": True,
            "cache_path": cache_path,
        }

        out_path = build_output_path(args.dataset_name, args.split, args.output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Loaded cached labels: {cache_path}")
        print(f"Saved: {out_path}")
        print("Selected labels:")
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(backend["api_key_env"], "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing API key in env var: {backend['api_key_env']}")
    log_step(f"api key found in env var {backend['api_key_env']}")

    votes: List[Dict[str, Dict[str, object]]] = []
    for i in range(args.votes):
        vote_ok = False
        for parse_attempt in range(1, args.max_retries + 1):
            log_step(
                f"vote {i + 1}/{args.votes}: sending request (parse attempt {parse_attempt}/{args.max_retries})"
            )
            content = call_llm_completion(
                provider=args.provider,
                api_base=backend["api_base"],
                api_key=api_key,
                model=backend["model"],
                prompt=prompt,
                temperature=args.temperature,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            log_step(f"vote {i + 1}/{args.votes}: response received")
            log_step(f"vote {i + 1}/{args.votes}: parsing json")
            try:
                obj = extract_json_obj(content)
                votes.append(normalize_vote_result(obj, cfg.attributes))
                vote_ok = True
                log_step(f"vote {i + 1}/{args.votes}: done")
                break
            except ValueError as e:
                log_step(f"vote {i + 1}/{args.votes}: parse failed ({e})")
                if parse_attempt < args.max_retries:
                    time.sleep(args.retry_sleep)

        if not vote_ok:
            raise RuntimeError(
                f"Failed to parse JSON for vote {i + 1} after {args.max_retries} attempts. "
                "可尝试降低 --num_pairs 或更换模型。"
            )

    log_step("aggregating votes")
    final = aggregate_votes(votes, cfg.attributes)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for attr, item in final.items():
        final_label = str(item.get("final_label", "uncertain"))
        grouped[final_label].append(attr)
    selected_labels = build_selected_labels(final)
    log_step("saving cache and report")

    save_selected_labels(
        cache_path=cache_path,
        provider=args.provider,
        model=backend["model"],
        dataset_name=args.dataset_name,
        split=args.split,
        selected=selected_labels,
        final_by_attribute=final,
    )

    # 输出聚合结果与可复用标签文件。
    report = {
        "dataset": args.dataset_name,
        "split": args.split,
        "source_file": split_file,
        "num_pairs_used": len(pair_examples),
        "votes": args.votes,
        "provider": args.provider,
        "model": backend["model"],
        "api_base": backend["api_base"],
        "attributes": cfg.attributes,
        "schema_summary": schema_summary,
        "final_by_attribute": final,
        "grouped_labels": dict(grouped),
        "selected_labels": selected_labels,
        "pair_examples": pair_examples,
        "used_cached_labels": False,
        "cache_path": cache_path,
    }

    out_path = build_output_path(args.dataset_name, args.split, args.output_file)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print(f"Saved label cache: {cache_path}")
    log_step("finished")
    print("Final grouped labels:")
    print(json.dumps(report["grouped_labels"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
