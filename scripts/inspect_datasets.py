"""Inspect the locally pinned evaluation datasets without downloading anything."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(os.environ.get("RL_FAILURES_ROOT", Path(__file__).resolve().parents[1])) / "data"


def parquet_summary(path: Path) -> None:
    table = pq.read_table(path)
    row = table.slice(0, 1).to_pylist()[0]
    print(f"\n{path}")
    print(f"rows={table.num_rows}")
    print(f"columns={table.column_names}")
    print(f"sample_keys={list(row)}")
    print(f"sample_types={ {key: type(value).__name__ for key, value in row.items()} }")
    if "chosen" in row:
        print(f"chosen_message_roles={[message.get('role') for message in row['chosen']]}")
        print(f"chosen_message_keys={[list(message) for message in row['chosen']]}")


def main() -> None:
    parquet_summary(ROOT / "ultrafeedback_binarized/data/test-00000-of-00001.parquet")
    parquet_summary(ROOT / "gsm8k/main/test-00000-of-00001.parquet")

    path = ROOT / "ifeval/ifeval_input_data.jsonl"
    with path.open(encoding="utf-8") as handle:
        row = json.loads(next(handle))
    print(f"\n{path}")
    print(f"sample_keys={list(row)}")
    print(f"sample_types={ {key: type(value).__name__ for key, value in row.items()} }")


if __name__ == "__main__":
    main()
