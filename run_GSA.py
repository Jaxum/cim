import os
import glob
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    LineByLineTextDataset, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from src.config import DATASET_METADATA


def _resolve_dataset_config(dataset_name: str):
    """Resolve dataset config by name/alias.

    Supports:
    1) direct registry key (e.g., "Magellan_Amazon_Google")
    2) exact DatasetConfig.name match
    3) data_path tail match (case/underscore/dash insensitive)
    """
    from src.config import get_dataset_config, DATASET_METADATA

    # 1) Direct key match
    if dataset_name in DATASET_METADATA:
        return DATASET_METADATA[dataset_name]

    # 2) Match the name field
    for cfg in DATASET_METADATA.values():
        if dataset_name == cfg.name:
            return cfg

    # 3) Match the data_path tail (case/symbol tolerant)
    norm = lambda s: s.replace('_', '').replace('-', '').lower()
    wanted = norm(dataset_name)
    for cfg in DATASET_METADATA.values():
        tail = os.path.basename(cfg.data_path)
        if norm(tail) == wanted:
            return cfg

    raise ValueError(f"Unable to resolve dataset name: {dataset_name}. Options: {list(DATASET_METADATA.keys())}")


def _select_corpus_files(files, corpus_splits: str):
    corpus_splits = (corpus_splits or "all").lower().strip()
    if corpus_splits not in {"train", "train_valid", "all"}:
        raise ValueError(f"Unknown corpus_splits: {corpus_splits}")

    wanted_prefixes = ["train.txt"]
    if corpus_splits in {"train_valid", "all"}:
        wanted_prefixes.append("valid.txt")
    if corpus_splits == "all":
        wanted_prefixes.append("test.txt")

    selected = []
    for p in files:
        base = os.path.basename(p).lower()
        # WDC-compatible naming: train.txt.medium / valid.txt.small
        if any(base.startswith(prefix) for prefix in wanted_prefixes):
            selected.append(p)
    return selected


def aggregate_corpus(dataset_name, output_file="gsa_corpus.txt", corpus_splits: str = "all"):
    """
    Scan the dataset, extract text fields, and build a corpus file.
    """
    print(f"Building GSA corpus for {dataset_name}: {output_file} ...")
    
    lines = []
    
    # Resolve dataset config
    try:
        dataset_config = _resolve_dataset_config(dataset_name)
        data_path = os.path.join("data", dataset_config.data_path)
    except ValueError:
        print(f"Dataset config not found for {dataset_name}; scanning all files under data/.")
        data_path = "data"

    files = glob.glob(f"{data_path}/**/*.txt", recursive=True)
    split_files = _select_corpus_files(files, corpus_splits=corpus_splits)
    if split_files:
        files = split_files
        print(f"Corpus splits={corpus_splits}; using {len(files)} files")
    else:
        print(
            f"Warning: no files matched splits={corpus_splits} under {data_path}; using all txt files: {len(files)}"
        )
    
    if not files:
        print(f"Warning: no txt files found under {data_path}.")
        return None

    for file_path in files:
        if "README" in file_path:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        # Left and right entity strings
                        lines.append(parts[0])
                        lines.append(parts[1])
        except Exception as e:
            print(f"Skip file {file_path}: {e}")

    # Deduplicate and write
    unique_lines = list(set(lines))
    
    # Ensure output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + "\n")
            
    print(f"Corpus built: {len(unique_lines)} unique lines.")
    return output_file

def get_gsa_output_dir(dataset_name):
    """
    Build the GSA output directory name from the dataset name.
    Dirty datasets use the Dirty_ prefix to separate them from Structured datasets.
    """
    try:
        cfg = _resolve_dataset_config(dataset_name)
        base_name = os.path.basename(cfg.data_path)
        if "Dirty" in cfg.data_path:
            short_name = f"Dirty_{base_name}"
        else:
            short_name = base_name
    except Exception:
        short_name = dataset_name
    return os.path.join("GSA", short_name)


def run_gsa(
    dataset_name,
    model_name="roberta-base",
    output_dir=None,
    epochs=5,
    batch_size=16,
    lr=2e-5,
    corpus_splits: str = "all",
):
    """
    Run masked language model training for GSA.
    """
    # Resolve dataset and default output directory: GSA/<dataset-short>
    if output_dir is None:
        output_dir = get_gsa_output_dir(dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    # Corpus file path
    corpus_file = os.path.join(output_dir, "corpus.txt")
    
    # 1) Build corpus for this dataset
    if not aggregate_corpus(dataset_name, corpus_file, corpus_splits=corpus_splits):
        print("Failed to build corpus; GSA aborted.")
        return

    print(f"Start GSA training. Base model: {model_name}")
    print(f"Output dir: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=corpus_file,
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,  
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        learning_rate=lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GSA complete. Model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or alias (e.g., Amazon-Google)")
    parser.add_argument("--base_model", type=str, default="roberta-base", help="Base model path or name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: GSA/<dataset>)")
    parser.add_argument("--epochs", type=int, default=15, help="GSA training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--corpus_splits",
        type=str,
        default="all",
        choices=["train", "train_valid", "all"],
        help="Corpus splits: all (train+valid+test), train_valid, or train",
    )
    args = parser.parse_args()

    # Prefer a local roberta-base if available
    base_model = args.base_model
    if os.path.exists("d:/causal_er/roberta-base") and base_model == "roberta-base":
        base_model = "d:/causal_er/roberta-base"

    # Resolve default output directory
    out_dir = args.output_dir or get_gsa_output_dir(args.dataset_name)


    run_gsa(
        args.dataset_name,
        model_name=base_model,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        corpus_splits=args.corpus_splits,
    )
