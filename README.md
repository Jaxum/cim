# CIM: Causal Entity Matching

This repo contains a CIM (Causal Entity Matching) framework for entity resolution with attribute-level causal gating and optional GSA pretraining.

## Project Layout

- main.py: training entrypoint
- matcher.py: inference/evaluation entrypoint
- run_tapt.py: GSA pretraining
- src/: core model, dataset, trainer, and config
- data/: datasets (see src/config.py for dataset metadata)
- saved_models/: training outputs

## Setup

1) Create a Python environment (3.9+ recommended).
2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Training is configured via src/config.py. CLI is intentionally minimal; use config defaults for most settings.

Example:

```bash
python main.py --dataset_name Magellan_Amazon_Google --data_dir ./data --output_dir ./saved_models
```

Optional flags:

- --encoder_model_name: override pretrained encoder path/name
- --max_len: override tokenizer max length
- --epochs, --lr, --train_batch_size, --eval_batch_size, --gradient_accumulation_steps
- --seed, --device, --use_wandb, --oversample_train
- --enable_gsa: run GSA automatically if missing
- --re_gsa: force rerun GSA even if it exists
- --use_fp16 / --disable_fp16

## GSA Pretraining

Run GSA directly if you want to control hyperparameters:

```bash
python run_tapt.py --dataset_name Magellan_Amazon_Google --base_model roberta-base --epochs 15 --batch_size 16 --lr 5e-5
```

GSA outputs are saved under GSA/<dataset> by default.

## Inference / Evaluation

Use matcher.py to run inference on the test split and report metrics. Threshold defaults to 0.5.

```bash
python matcher.py --checkpoint_path ./saved_models/Magellan_Amazon_Google/best_model.pt --dataset_name Magellan_Amazon_Google --data_dir ./data
```

Optional flags:

- --batch_size: inference batch size
- --threshold: classification threshold
- --output_file: save predictions and metrics JSON

## Notes

- Dataset metadata (attributes, causal/confounder fields) is defined in src/config.py.
- WDC datasets use train_size-specific files (train.txt.<size>), but size selection is handled in src/dataset.py.
