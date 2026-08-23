from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


LOGS_STAGE_ROOT = Path("/data/dn/FRTP_revision1/imagecls/logs_stage")
RECONS_METRICS = ("relative_diff", "absolute_diff", "forward_norm")


@dataclass(frozen=True)
class RecoverySpec:
    log_path: Path
    model: str
    lr: float
    expected_epochs: int
    recons_keys: list[str]

    @property
    def save_path(self) -> Path:
        return LOGS_STAGE_ROOT / f"{self.model}_{self.lr}lr.pt"


RECOVERY_SPECS = {
    "tin": RecoverySpec(
        log_path=LOGS_STAGE_ROOT / "tin_output.log",
        model="tin_simplecnnv2_relu",
        lr=1e-5,
        expected_epochs=100,
        recons_keys=[
            "recons_out",
            "recons_z5",
            "recons_z4",
            "recons_z3",
            "recons_z2",
            "recons_z1",
        ],
    ),
    "nette": RecoverySpec(
        log_path=LOGS_STAGE_ROOT / "nette_output.log",
        model="nette_simplecnnv2_relu",
        lr=1e-5,
        expected_epochs=100,
        recons_keys=[
            "recons_out",
            "recons_z6",
            "recons_z5",
            "recons_z4",
            "recons_z3",
            "recons_z2",
            "recons_z1",
        ],
    ),
}


def compact_log_text(raw_text: str) -> str:
    """Undo tmux hard wrapping and drop noisy lines printed during metric collection."""
    kept_lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "[reliable conv] some value exceeds 100.0":
            continue
        if stripped.startswith("(/data/conda_envs/"):
            continue
        if stripped == "clear":
            continue
        kept_lines.append(line.rstrip("\n"))
    return "".join(kept_lines)


def find_matching_brace(text: str, start: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"unmatched brace starting at character {start}")


def extract_dict_after(text: str, label: str, start: int) -> tuple[dict[str, Any], int]:
    label_pos = text.find(label, start)
    if label_pos < 0:
        raise ValueError(f"missing {label!r} after character {start}")
    brace_start = text.find("{", label_pos)
    if brace_start < 0:
        raise ValueError(f"missing dict for {label!r} after character {label_pos}")
    brace_end = find_matching_brace(text, brace_start)
    return ast.literal_eval(text[brace_start : brace_end + 1]), brace_end + 1


def init_recons_history(recons_keys: list[str]) -> dict[str, dict[str, list[float]]]:
    return {key: {metric: [] for metric in RECONS_METRICS} for key in recons_keys}


def append_recons_history(
    history: dict[str, dict[str, list[float]]],
    values: dict[str, dict[str, float]],
) -> None:
    for recons_key in history:
        if recons_key not in values:
            raise ValueError(f"missing recons key {recons_key!r} in parsed epoch")
        for metric in RECONS_METRICS:
            history[recons_key][metric].append(float(values[recons_key][metric]))


def recover_results(spec: RecoverySpec) -> dict[str, Any]:
    text = compact_log_text(spec.log_path.read_text())
    matches = list(re.finditer(r"Epoch \[(\d+)/(\d+)\] train_score=([0-9.]+), test_score=([0-9.]+),", text))
    selected = [match for match in matches if int(match.group(2)) == spec.expected_epochs]
    if not selected:
        raise ValueError(f"found no {spec.expected_epochs}-epoch records in {spec.log_path}")

    train_score_list: list[float] = []
    test_score_list: list[float] = []
    train_recons_loss_list = init_recons_history(spec.recons_keys)
    test_recons_loss_list = init_recons_history(spec.recons_keys)
    seen_epochs: list[int] = []

    for match in selected:
        epoch = int(match.group(1))
        seen_epochs.append(epoch)
        train_score_list.append(float(match.group(3)))
        test_score_list.append(float(match.group(4)))
        train_stats, cursor = extract_dict_after(text, "train_recons_stats=", match.end())
        test_stats, _ = extract_dict_after(text, "test_recons_stats=", cursor)
        append_recons_history(train_recons_loss_list, train_stats)
        append_recons_history(test_recons_loss_list, test_stats)

    expected_record_count = len(range(0, spec.expected_epochs, 5))
    if len(selected) != expected_record_count:
        raise ValueError(
            f"{spec.log_path} yielded {len(selected)} records, expected {expected_record_count}; "
            f"epochs={seen_epochs}"
        )

    return {
        "model": spec.model,
        "lr": spec.lr,
        "num_epochs": spec.expected_epochs,
        "recons_keys": spec.recons_keys,
        "recons_metrics": RECONS_METRICS,
        "train_score_list": train_score_list,
        "test_score_list": test_score_list,
        "train_recons_loss_list": train_recons_loss_list,
        "test_recons_loss_list": test_recons_loss_list,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover different_stage.py result .pt files from tmux logs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(RECOVERY_SPECS),
        default=tuple(RECOVERY_SPECS),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing recovered .pt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.experiments:
        spec = RECOVERY_SPECS[name]
        results = recover_results(spec)
        if spec.save_path.exists() and not args.force:
            raise FileExistsError(f"{spec.save_path} already exists; pass --force to overwrite")
        torch.save(results, spec.save_path)
        print(
            f"saved {name}: {spec.save_path} "
            f"({len(results['train_score_list'])} records, epochs={results['num_epochs']})"
        )


if __name__ == "__main__":
    main()
