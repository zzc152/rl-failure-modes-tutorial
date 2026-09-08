"""Score already-generated base-model responses without rerunning inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


PROJECT = Path("/workspace/zzc/rl-failures")
RESULTS = PROJECT / "results/base_model"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    config = yaml.safe_load((PROJECT / "configs/experiment_constants.yaml").read_text(encoding="utf-8"))
    preference_scores = json.loads((RESULTS / "uf_rank0.json").read_text(encoding="utf-8"))
    ifeval_outputs = read_jsonl(RESULTS / "ifeval_responses.jsonl")
    gsm_outputs = read_jsonl(RESULTS / "gsm8k_responses.jsonl")

    sys.path.insert(0, str(PROJECT / "third_party/google-research"))
    from instruction_following_eval import evaluation_lib  # pylint: disable=import-outside-toplevel

    ifeval_inputs = evaluation_lib.read_prompt_list(str(Path(config["data"]["ifeval"]["path"])))
    responses = {row["prompt"]: row["response"] for row in ifeval_outputs}
    strict = [evaluation_lib.test_instruction_following_strict(item, responses) for item in ifeval_inputs]
    write_jsonl(
        RESULTS / "ifeval_strict_details.jsonl",
        [{"follow_all_instructions": item.follow_all_instructions} for item in strict],
    )
    metrics = {
        "ultrafeedback_preference_accuracy": sum(chosen > rejected for chosen, rejected in preference_scores) / len(preference_scores),
        "ultrafeedback_valid_pairs": len(preference_scores),
        "ifeval_strict": sum(item.follow_all_instructions for item in strict) / len(strict),
        "gsm8k_accuracy": sum(row["prediction"] == row["reference"] for row in gsm_outputs) / len(gsm_outputs),
        "avg_response_length": sum(len(row["response"].split()) for row in ifeval_outputs + gsm_outputs) / (len(ifeval_outputs) + len(gsm_outputs)),
        "constants": config,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
