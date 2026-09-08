"""Train the fixed small clean-DPO baseline on UltraFeedback Binarized."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


PROJECT = Path(os.environ.get("RL_FAILURES_ROOT", Path(__file__).resolve().parents[1]))


def main() -> None:
    config = yaml.safe_load((PROJECT / "configs/dpo_clean.yaml").read_text(encoding="utf-8"))
    seed = config["dataset"]["seed"]
    set_seed(seed)
    model_path = PROJECT / "models/Qwen2.5-1.5B-Instruct"
    data_path = PROJECT / "data/ultrafeedback_binarized/data/train-00000-of-00001.parquet"
    output_dir = PROJECT / "results/dpo_clean"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def format_pair(row: dict) -> dict:
        prompt = tokenizer.apply_chat_template(
            row["chosen"][:-1], tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt,
            "chosen": row["chosen"][-1]["content"],
            "rejected": row["rejected"][-1]["content"],
        }

    dataset = Dataset.from_parquet(str(data_path))
    dataset = dataset.shuffle(seed=seed).select(range(config["dataset"]["train_examples"]))
    dataset = dataset.map(format_pair, remove_columns=dataset.column_names)
    policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
    reference = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
    settings = config["dpo"]
    args = DPOConfig(
        output_dir=str(output_dir),
        beta=settings["beta"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        max_length=settings["max_length"],
        bf16=settings["bf16"],
        gradient_checkpointing=True,
        num_train_epochs=settings["num_train_epochs"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))


if __name__ == "__main__":
    main()
