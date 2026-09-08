"""Small, CPU-only health check for UltraFeedback chat-token construction."""

import os
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


PROJECT = Path(os.environ.get("RL_FAILURES_ROOT", Path(__file__).resolve().parents[1]))
MODEL = PROJECT / "models/Qwen2.5-1.5B-Instruct"
DATA = PROJECT / "data/ultrafeedback_binarized/data/test-00000-of-00001.parquet"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = pq.read_table(DATA).slice(0, 10).to_pylist()
    valid = 0
    for index, row in enumerate(rows):
        if index == 0:
            user_content = row["chosen"][0]["content"]
            print(f"user_content_type={type(user_content).__name__}")
            print(f"user_content_preview={repr(user_content)[:500]}")
        prompt_ids = tokenizer.apply_chat_template(
            row["chosen"][:-1], tokenize=True, add_generation_prompt=True
        )
        if index == 0:
            print(f"prompt_ids_type={type(prompt_ids).__name__}")
            print(f"prompt_ids_repr={repr(prompt_ids)[:500]}")
            print(f"prompt_decoded={repr(tokenizer.decode(prompt_ids))[:500]}")
        chosen = tokenizer(row["chosen"][-1]["content"], add_special_tokens=False).input_ids
        rejected = tokenizer(row["rejected"][-1]["content"], add_special_tokens=False).input_ids
        is_valid = bool(prompt_ids and chosen and rejected)
        valid += is_valid
        print(index, len(prompt_ids), len(chosen), len(rejected), is_valid)
    print(f"valid_pairs={valid}/{len(rows)}")


if __name__ == "__main__":
    main()
