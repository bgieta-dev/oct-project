"""Runs several pipeline configs one after another as consecutive experiments (testN, testN+1, ...).

Each config runs in its own `python -m octseg.pipeline --promote` subprocess (fresh CUDA context per run).
Before any training, every config is loaded and RETFound weights are fetched/validated, so a missing
HF token or a typo fails in seconds instead of after hours. A failed run does not stop the series,
unless the GPU itself is gone (nvidia-smi fails), in which case the remaining runs are skipped.

Usage:
    python -m octseg.run_series --configs configs/retfound_lora.yaml configs/retfound_lora_nodetail.yaml configs/retfound_frozen.yaml
    python -m octseg.run_series --configs ... --epochs 1     # quick smoke of the whole series
    python -m octseg.run_series --configs ... --push         # git add/commit/push experiments/ at the end
"""
import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

from octseg.config import load_config
from octseg.runs import ROOT, get_next_experiment_name, make_run_dir, send_discord_notification, setup_logging

log = logging.getLogger(__name__)


def preflight(config_paths):
    """Loads every config and, for ARCH=retfound with pretrained weights, downloads and validates them once."""
    checked_weights = set()
    for path in config_paths:
        cfg = load_config(path)
        log.info(f"Config OK: {path} (ARCH={cfg.ARCH})")
        if cfg.ARCH == "retfound" and cfg.RETFOUND_WEIGHTS and cfg.RETFOUND_WEIGHTS not in checked_weights:
            from octseg.model import build_retfound
            build_retfound(cfg)
            checked_weights.add(cfg.RETFOUND_WEIGHTS)


def gpu_lost() -> bool:
    """True if nvidia-smi exists but fails (e.g. 'GPU is lost'); False when there is no NVIDIA tooling."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    result = subprocess.run([smi], capture_output=True, text=True, check=False)
    return result.returncode != 0 or "GPU is lost" in result.stdout + result.stderr


def read_metrics(exp_name):
    """metrics.yaml of a promoted experiment, or None if the run produced no evaluation."""
    path = ROOT / "experiments" / exp_name / "metrics.yaml"
    return yaml.safe_load(path.read_text()) if path.is_file() else None


def format_summary(rows):
    lines = ["| experiment | config | status | mIoU | mDice | mHD95 | IRF Dice | SRF Dice | PED Dice |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, cfg_path, status, m in rows:
        if m:
            dice = {int(k): v for k, v in m["class_dices"].items()}
            vals = [f"{m['mIoU']:.4f}", f"{m['mDice']:.4f}", f"{m['mHD95']:.2f}",
                    f"{dice[1]:.4f}", f"{dice[2]:.4f}", f"{dice[3]:.4f}"]
        else:
            vals = ["-"] * 6
        lines.append(f"| {name} | {cfg_path} | {status} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def push_results(names):
    msg = f"results: {', '.join(names)}"
    for cmd in (["git", "add", "experiments"], ["git", "commit", "-m", msg],
                ["git", "pull", "--rebase"], ["git", "push"]):
        if subprocess.run(cmd, cwd=ROOT, check=False).returncode != 0:
            log.error(f"Git step failed: {' '.join(cmd)}. Push manually with scripts/git_update.")
            return
    log.info("Results pushed.")


def main():
    parser = argparse.ArgumentParser(description="Run configs sequentially as consecutive testN experiments")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML configs, run in the given order")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for every run")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip config/weights check")
    parser.add_argument("--push", action="store_true", help="git add/commit/push experiments/ when done")
    args = parser.parse_args()

    series_dir = make_run_dir("series", "series")
    setup_logging(series_dir, "series.log")
    log.info(f"Series of {len(args.configs)} runs: {args.configs}")

    if not args.skip_preflight:
        try:
            preflight(args.configs)
        except Exception as e:
            log.error(f"Preflight failed, nothing was trained: {e}")
            sys.exit(1)

    rows = []
    for i, cfg_path in enumerate(args.configs):
        name = get_next_experiment_name()
        log.info(f"=== [{i + 1}/{len(args.configs)}] {name}: {cfg_path} ({datetime.now():%Y-%m-%d %H:%M}) ===")
        cmd = [sys.executable, "-m", "octseg.pipeline", "--promote", "--config", cfg_path, "--name", name]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        subprocess.run(cmd, cwd=ROOT, check=False)
        metrics = read_metrics(name)
        status = "ok" if metrics else "FAILED"
        rows.append((name, cfg_path, status, metrics))
        log.info(f"{name}: {status}")
        if not metrics and gpu_lost():
            remaining = args.configs[i + 1:]
            for skipped in remaining:
                rows.append(("-", skipped, "SKIPPED (GPU lost)", None))
            log.error("GPU unavailable (nvidia-smi fails). Run scripts\\recover_gpu.bat (or reboot), then rerun: "
                      f"--configs {' '.join(remaining)}")
            break

    summary = format_summary(rows)
    log.info("Series summary:\n" + summary)
    (series_dir / "summary.md").write_text(summary + "\n")
    done = [r[0] for r in rows if r[2] == "ok"]
    if done:
        (ROOT / "experiments" / f"series_{done[0]}-{done[-1]}.md").write_text(summary + "\n")
    send_discord_notification(f"**OCT series finished**\n```\n{summary}\n```")

    if args.push and done:
        push_results(done)
    sys.exit(0 if all(r[2] == "ok" for r in rows) else 1)


if __name__ == "__main__":
    main()
