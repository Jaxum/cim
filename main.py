import argparse
import json
import random
import numpy as np
import torch
import os
import sys

from src.config import MainConfig, ModelConfig, get_dataset_config, get_full_config
from src.dataset import get_dataloaders
from src.model import initialize_model
from src.trainer import Trainer

# Import GSA tool
import subprocess


def set_seed(seed: int):
    """
    设置随机种子以确保可复现性。
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    """
    解析命令行参数。
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="因果实体解析 (CIM) 框架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 主配置
    parser.add_argument("--dataset_name", type=str, default="WDC_Computers", help="要使用的数据集名称")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./saved_models", help="保存模型检查点的目录")
    
    # 模型配置（为 None 时使用 config.py 中的默认值）
    parser.add_argument("--encoder_model_name", type=str, default=None, help="预训练编码器路径或 Hugging Face 模型名称")
    parser.add_argument("--max_len", type=int, default=None, help="分词器的最大序列长度")
    
    # 训练配置（为 None 时使用 config.py 中的默认值）
    parser.add_argument("--epochs", type=int, default=None, help="训练的轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--train_batch_size", type=int, default=None, help="训练批量大小")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="评估批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="梯度累积步数（有效 batch = train_batch_size * gradient_accumulation_steps）")
    
    # 其他设置
    parser.add_argument("--seed", type=int, default=123, help="随机种子（用于可复现性），设为 -1 则使用随机种子")
    parser.add_argument("--use_wandb", action="store_true", help="使用 Weights & Biases 进行实验跟踪")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备：cuda 或 cpu")
    parser.add_argument("--oversample_train", action="store_true", help="训练时对少数类进行过采样")

    # 混合精度（默认使用 config.py 的 ModelConfig.use_fp16）
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--use_fp16", action="store_true", help="显式启用 FP16 混合精度（仅 cuda 生效）")
    fp16_group.add_argument("--disable_fp16", action="store_true", help="显式关闭 FP16 混合精度")
    
    # GSA settings
    parser.add_argument("--enable_gsa", action="store_true", help="Enable GSA pretraining")
    # Note: GSA hyperparameters (lr/batch_size/epochs) are configured in run_GSA.py.
    # If you need custom settings, run run_GSA.py directly.
    parser.add_argument("--re_gsa", action="store_true", help="Force rerun GSA even if a model exists")

    # 自动标注设置
    parser.add_argument("--llm_attr_auto", action="store_true", help="Auto-select causal/confounder attrs via LLM")
    parser.add_argument("--llm_attr_provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--llm_attr_model", default=None, help="Override LLM model name")
    parser.add_argument("--llm_attr_api_base", default=None, help="Override LLM API base URL")
    parser.add_argument("--llm_attr_api_key_env", default=None, help="Override LLM API key env var")
    parser.add_argument("--llm_attr_split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--llm_attr_num_pairs", type=int, default=8)
    parser.add_argument("--llm_attr_votes", type=int, default=3)
    parser.add_argument("--llm_attr_reuse_cached", action="store_true")
    parser.add_argument("--llm_attr_refresh_cached", action="store_true")
    
    
    return parser.parse_args()


def run_gsa_if_needed(dataset_name, base_model, args):
    """
    Check whether a GSA model exists; run GSA if missing.
    
    返回:
        str: GSA model path if enabled; otherwise the base model
    """
    # Resolve dataset short name (must match run_GSA.py logic)
    try:
        dataset_config = get_dataset_config(dataset_name)
        base_name = os.path.basename(dataset_config.data_path)
        if "Dirty" in dataset_config.data_path:
            short_name = f"Dirty_{base_name}"
        else:
            short_name = base_name
    except Exception:
        short_name = dataset_name
    
    gsa_dir = os.path.join("GSA", short_name)
    
    # Check whether a GSA model exists
    gsa_exists = os.path.exists(gsa_dir) and os.path.exists(os.path.join(gsa_dir, "config.json"))
    
    if args.re_gsa or not gsa_exists:
        print("\n" + "="*80)
        print("Step 1: GSA")
        print("="*80)
        
        if args.re_gsa:
            print(f"Force rerun GSA: {gsa_dir}")
        else:
            print(f"No GSA model found: {gsa_dir}")
            print("Starting GSA training...")
        
        print(f"Dataset: {dataset_name}")
        print(f"Base model: {base_model}")
        print("GSA will use defaults from run_GSA.py.")
        print()
        
        # Call run_GSA.py
        cmd = [
            sys.executable,  # Use current Python interpreter
            "run_GSA.py",
            "--dataset_name", dataset_name,
            "--base_model", base_model,
            "--corpus_splits", "all",
        ]
        
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=False)
            print(f"\nGSA complete. Model saved to: {gsa_dir}")
            print("="*80 + "\n")
            return gsa_dir
        except subprocess.CalledProcessError as e:
            print(f"\nGSA failed: {e}")
            print("Falling back to the base model...")
            print("="*80 + "\n")
            return base_model
    else:
        print(f"\n✓ Found existing GSA model: {gsa_dir}")
        print("Loading GSA weights for training\n")
        return gsa_dir


def main():
    """
    主函数：协调整个工作流程。
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("因果实体解析 (CIM) 框架")
    print("="*80)
    print(f"数据集: {args.dataset_name}")
    print(f"设备: {args.device}")
    if args.enable_gsa:
        print("GSA: enabled (auto-detect/run)")
    print("="*80 + "\n")
    
    # 随机种子
    if args.seed == -1:
        # 使用随机种子
        import time
        args.seed = int(time.time() * 1000) % (2**31)
        print(f"使用随机种子: {args.seed}")
    set_seed(args.seed)
    print(f"随机种子设置为: {args.seed}\n")
    

    # 自动将数据集名称附加到输出目录
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    
    main_config = MainConfig(
        dataset_name=args.dataset_name,
        run_mode="train",
        output_dir=output_dir,
        data_dir=args.data_dir
    )
    
    # 加载 config.py 的默认配置
    model_config = ModelConfig()
    
    # GSA: auto-check and run if enabled
    base_encoder = args.encoder_model_name if args.encoder_model_name is not None else model_config.encoder_model_name
    
    if args.enable_gsa:
        encoder_to_use = run_gsa_if_needed(args.dataset_name, base_encoder, args)
        model_config.encoder_model_name = encoder_to_use
    elif args.encoder_model_name is not None:
        model_config.encoder_model_name = args.encoder_model_name
    
    # 仅在命令行提供参数时覆盖默认配置
    if args.max_len is not None:
        model_config.max_len = args.max_len
    if args.epochs is not None:
        model_config.epochs = args.epochs
    if args.lr is not None:
        model_config.lr = args.lr
    if args.train_batch_size is not None:
        model_config.train_batch_size = args.train_batch_size
    if args.eval_batch_size is not None:
        model_config.eval_batch_size = args.eval_batch_size
    if args.gradient_accumulation_steps is not None:
        model_config.gradient_accumulation_steps = args.gradient_accumulation_steps

    # FP16 覆盖（最后应用，确保不被其他逻辑误改）
    if getattr(args, "disable_fp16", False):
        model_config.use_fp16 = False
    elif getattr(args, "use_fp16", False):
        model_config.use_fp16 = True
    main_config.oversample_train = args.oversample_train
    
    model_config.seed = args.seed
    
    dataset_config = get_dataset_config(args.dataset_name)

    if args.llm_attr_auto:
        llm_out_dir = os.path.join("./attr_auto", "llm_labels")
        os.makedirs(llm_out_dir, exist_ok=True)
        llm_out_path = os.path.join(
            llm_out_dir, f"{args.dataset_name}_{args.llm_attr_split}_llm_labels.json"
        )

        cmd = [
            sys.executable,
            "llm_attribute_auto_label.py",
            "--dataset_name",
            args.dataset_name,
            "--data_root",
            args.data_dir,
            "--split",
            args.llm_attr_split,
            "--num_pairs",
            str(args.llm_attr_num_pairs),
            "--votes",
            str(args.llm_attr_votes),
            "--provider",
            args.llm_attr_provider,
            "--output_file",
            llm_out_path,
        ]

        if args.llm_attr_model:
            cmd.extend(["--model", args.llm_attr_model])
        if args.llm_attr_api_base:
            cmd.extend(["--api_base", args.llm_attr_api_base])
        if args.llm_attr_api_key_env:
            cmd.extend(["--api_key_env", args.llm_attr_api_key_env])
        if args.llm_attr_reuse_cached:
            cmd.append("--reuse_cached_labels")
        if args.llm_attr_refresh_cached:
            cmd.append("--refresh_cached_labels")

        print("Running LLM attribute labeling...")
        try:
            subprocess.run(cmd, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"LLM labeling failed: {e}")
            sys.exit(1)

        with open(llm_out_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        selected = report.get("selected_labels", {})
        if not isinstance(selected, dict):
            raise ValueError("Invalid LLM label report: selected_labels missing or invalid")

        dataset_config.causal_attributes = list(selected.get("causal", []))
        dataset_config.confounder_attributes = list(selected.get("confounder", []))
    
    
    # 创建数据加载器
    print("加载数据集 ...")
    try:
        train_loader, valid_loader, _ = get_dataloaders(
            main_config=main_config,
            model_config=model_config,
            dataset_config=dataset_config
        )
        print(f"✓ 数据加载器创建成功")
        print(f"  - 训练批次数: {len(train_loader)}")
        print(f"  - 验证批次数: {len(valid_loader)}\n")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请确保数据目录存在: {main_config.data_dir}")
        sys.exit(1)
    
    # 初始化模型
    print("初始化模型...")
    model = initialize_model(
        model_config=model_config,
        dataset_config=dataset_config,
        device=args.device
    )
    print()
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        main_config=main_config,
        model_config=model_config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=None,
        use_wandb=args.use_wandb
    )
    print()
    
    # 开始训练
    print("开始训练...\n")
    trainer.train()


if __name__ == "__main__":
    main()
