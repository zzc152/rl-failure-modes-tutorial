"""Run the immutable, pre-DPO benchmark for the tutorial's initial policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(os.environ.get("RL_FAILURES_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_PATH = PROJECT / "configs/experiment_constants.yaml"


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def rank_info() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def shard(items: list[dict], rank: int, world_size: int) -> list[tuple[int, dict]]:
    return [(index, item) for index, item in enumerate(items) if index % world_size == rank]


def chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate(
    model, tokenizer, prompts: list[str], decoding: dict, device: torch.device, batch_size: int = 8
) -> list[str]:
    responses: list[str] = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        output = model.generate(
            **encoded,
            do_sample=decoding["do_sample"],
            max_new_tokens=decoding["max_new_tokens"],
            use_cache=decoding["use_cache"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output[:, encoded.input_ids.shape[1] :]
        responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return responses


@torch.inference_mode()
def conditional_logps(model, tokenizer, rows: list[dict], device: torch.device) -> list[tuple[float, float]]:
    """Return mean log p(completion | user prompt) for chosen and rejected."""
    token_pairs = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["chosen"][:-1], tokenize=True, add_generation_prompt=True
        )["input_ids"]
        chosen_completion = tokenizer(row["chosen"][-1]["content"], add_special_tokens=False).input_ids
        rejected_completion = tokenizer(row["rejected"][-1]["content"], add_special_tokens=False).input_ids
        if not chosen_completion or not rejected_completion:
            continue
        chosen_ids = prompt_ids + chosen_completion
        rejected_ids = prompt_ids + rejected_completion
        prompt_len = len(prompt_ids)
        token_pairs.append([(chosen_ids, prompt_len), (rejected_ids, prompt_len)])

    scores: list[tuple[float, float]] = []
    for start in range(0, len(token_pairs), 4):
        batch = token_pairs[start : start + 4]
        flattened = [item for pair in batch for item in pair]
        max_len = max(len(ids) for ids, _ in flattened)
        input_ids = torch.full(
            (len(flattened), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        offsets: list[int] = []
        for index, (ids, _) in enumerate(flattened):
            offset = max_len - len(ids)
            offsets.append(offset)
            input_ids[index, offset:] = torch.tensor(ids, device=device)
        logits = model(input_ids=input_ids).logits.log_softmax(dim=-1)
        values: list[float] = []
        for index, (ids, prompt_len) in enumerate(flattened):
            completion = torch.tensor(ids[prompt_len:], device=device)
            positions = torch.arange(
                offsets[index] + prompt_len - 1,
                offsets[index] + len(ids) - 1,
                device=device,
            )
            values.append(logits[index, positions, completion].mean().item())
        scores.extend(zip(values[::2], values[1::2]))
    return scores


def exact_number(text: str) -> str | None:
    markers = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if markers:
        return markers[-1].replace(",", "")
    values = re.findall(r"(?<!\w)-?[\d,]+(?:\.\d+)?", text)
    return values[-1].replace(",", "") if values else None


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Project-relative or absolute checkpoint path")
    parser.add_argument("--run-name", default="base_model")
    cli = parser.parse_args()
    rank, world_size, local_rank = rank_info()
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    random.seed(config["experiment"]["seed"])
    torch.manual_seed(config["experiment"]["seed"])

    results = project_path(config["reporting"]["output_dir"]) / cli.run_name
    results.mkdir(parents=True, exist_ok=True)
    model_path = project_path(cli.model_path or config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    uf_path = project_path(config["data"]["ultrafeedback"]["path"]) / config["data"]["ultrafeedback"]["test_file"]
    uf_rows = pq.read_table(uf_path).to_pylist()[: config["data"]["ultrafeedback"]["eval_subset_size"]]
    local_uf = [row for _, row in shard(uf_rows, rank, world_size)]
    preference_scores = conditional_logps(model, tokenizer, local_uf, device)
    with (results / f"uf_rank{rank}.json").open("w", encoding="utf-8") as handle:
        json.dump(preference_scores, handle)

    with project_path(config["data"]["ifeval"]["path"]).open(encoding="utf-8") as handle:
        ifeval_rows = [json.loads(line) for line in handle]
    local_ifeval = shard(ifeval_rows, rank, world_size)
    ifeval_prompts = [chat_prompt(tokenizer, row["prompt"]) for _, row in local_ifeval]
    ifeval_responses = generate(model, tokenizer, ifeval_prompts, config["decoding"], device)
    write_jsonl(
        results / f"ifeval_rank{rank}.jsonl",
        ({"index": index, "prompt": row["prompt"], "response": response}
         for (index, row), response in zip(local_ifeval, ifeval_responses)),
    )

    gsm_path = project_path(config["data"]["gsm8k"]["path"])
    gsm_rows = pq.read_table(gsm_path).to_pylist()
    sampled_indices = sorted(random.Random(config["experiment"]["seed"]).sample(
        range(len(gsm_rows)), config["data"]["gsm8k"]["eval_subset_size"]
    ))
    gsm_subset = [gsm_rows[index] for index in sampled_indices]
    local_gsm = shard(gsm_subset, rank, world_size)
    gsm_instruction = "Solve the problem step by step. End the response with `#### <number>`.\n\n"
    gsm_prompts = [chat_prompt(tokenizer, gsm_instruction + row["question"]) for _, row in local_gsm]
    gsm_responses = generate(model, tokenizer, gsm_prompts, config["decoding"], device)
    write_jsonl(
        results / f"gsm8k_rank{rank}.jsonl",
        ({"index": index, "question": row["question"], "reference": exact_number(row["answer"]),
          "prediction": exact_number(response), "response": response}
         for (index, row), response in zip(local_gsm, gsm_responses)),
    )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        preference_scores = []
        ifeval_outputs, gsm_outputs = [], []
        for worker in range(world_size):
            preference_scores.extend(json.loads((results / f"uf_rank{worker}.json").read_text(encoding="utf-8")))
            for filename, target in ((f"ifeval_rank{worker}.jsonl", ifeval_outputs), (f"gsm8k_rank{worker}.jsonl", gsm_outputs)):
                with (results / filename).open(encoding="utf-8") as handle:
                    target.extend(json.loads(line) for line in handle)
        ifeval_outputs.sort(key=lambda row: row["index"])
        gsm_outputs.sort(key=lambda row: row["index"])
        write_jsonl(results / "ifeval_responses.jsonl", ifeval_outputs)
        write_jsonl(results / "gsm8k_responses.jsonl", gsm_outputs)

        sys.path.insert(0, str(PROJECT / "third_party/google-research"))
        from instruction_following_eval import evaluation_lib  # pylint: disable=import-outside-toplevel
        ifeval_inputs = evaluation_lib.read_prompt_list(str(project_path(config["data"]["ifeval"]["path"])))
        responses = {row["prompt"]: row["response"] for row in ifeval_outputs}
        strict = [evaluation_lib.test_instruction_following_strict(item, responses) for item in ifeval_inputs]
        ifeval_strict = sum(item.follow_all_instructions for item in strict) / len(strict)
        write_jsonl(
            results / "ifeval_strict_details.jsonl",
            ({"follow_all_instructions": item.follow_all_instructions} for item in strict),
        )

        metrics = {
            "ultrafeedback_preference_accuracy": sum(chosen > rejected for chosen, rejected in preference_scores) / len(preference_scores),
            "ifeval_strict": ifeval_strict,
            "gsm8k_accuracy": sum(row["prediction"] == row["reference"] for row in gsm_outputs) / len(gsm_outputs),
            "avg_response_length": sum(len(row["response"].split()) for row in ifeval_outputs + gsm_outputs) / (len(ifeval_outputs) + len(gsm_outputs)),
            "constants": config,
        }
        (results / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
