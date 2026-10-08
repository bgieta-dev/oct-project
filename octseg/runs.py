"""Run-directory plumbing: logging, source/config snapshots, VRAM diagnostics, Discord notification."""
import re
from typing import Union
import logging
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def make_run_dir(kind, name="run", output=None):
    """Creates the run directory and returns its path.

    Default: ``outputs/<kind>/<timestamp>_<name>``; `output` overrides the whole path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output).resolve() if output else ROOT / "outputs" / kind / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def send_discord_notification(message):
    """Sends experiment status updates to the Discord webhook in `.env` (no-op if unset)."""
    load_dotenv(ROOT / ".env", override=True)
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message}, timeout=10)
    except Exception as e:
        log.error(f"Failed to send Discord notification: {e}")


def setup_logging(run_dir, filename="experiment.log"):
    """Configures the root logger with a verbose file handler and a short console handler.

    Every CLI entry point calls this once; modules log through ``logging.getLogger(__name__)``.
    """
    log_file = os.path.join(run_dir, filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    ))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    logging.captureWarnings(True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return log_file


def check_vram_diagnostics():
    """Reports VRAM usage prior to heavy training loops to catch likely out-of-memory states."""
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            log.info(f"VRAM Status Diagnostic: {free_gb:.2f} GB Free / {total_gb:.2f} GB Total")
            if free_gb < 2.0:
                log.warning("Extremely low VRAM detected (< 2.0 GB). Risk of training Out-Of-Memory failure is elevated.")
        except Exception as e:
            log.error(f"Could not retrieve precise VRAM information: {e}")


def archive_experiment_sources(run_dir):
    """Snapshots code, pinned requirements, splits and git state into the run directory.

    The resolved config is written separately (``config.yaml``) by each CLI.
    """
    log.info("--- SOURCE ARCHIVING ---")
    try:
        shutil.copytree(ROOT / "octseg", os.path.join(run_dir, "src"), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "requirements.txt", run_dir)
        shutil.copytree(ROOT / "splits", os.path.join(run_dir, "splits"))
        try:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
            with open(os.path.join(run_dir, "git.txt"), "w") as f:
                f.write(f"commit: {head}\nstatus (porcelain):\n{status}")
        except (OSError, subprocess.CalledProcessError):
            log.warning("git not available; git.txt not written.")
        log.info("Source code and configurations archiving complete.")
    except Exception as e:
        log.error(f"Error encountered during source archiving: {e}")
        log.error(traceback.format_exc())


def get_next_experiment_name(experiments_md_path=None) -> str:
    """Returns the next experiment name (e.g. 'test19') based on docs/EXPERIMENTS.md."""
    md_path = Path(experiments_md_path) if experiments_md_path else ROOT / "docs" / "EXPERIMENTS.md"
    max_num = 0
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        matches = re.findall(r"\|\s*test(\d+)\b", text)
        if matches:
            max_num = max(int(m) for m in matches)
    if max_num == 0:
        exp_dir = ROOT / "experiments"
        if exp_dir.exists():
            for entry in exp_dir.iterdir():
                m = re.match(r"^test(\d+)", entry.name)
                if m:
                    max_num = max(max_num, int(m.group(1)))
    return f"test{max_num + 1}"


def promote_run_to_experiment(run_dir: Union[str, Path], exp_name: str) -> Path:
    """Promotes run outputs into experiments/<exp_name>, strictly excluding checkpoints."""
    run_path = Path(run_dir).resolve()
    dest_path = ROOT / "experiments" / exp_name
    dest_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Promoting run {run_path.name} to {dest_path.relative_to(ROOT)}...")
    for item in run_path.iterdir():
        if item.name.endswith(".pth") or item.name.endswith(".pt"):
            log.info(f"Skipping checkpoint: {item.name}")
            continue
        dest_item = dest_path / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                dest_item,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.pth", "*.pt", "__pycache__"),
            )
        else:
            shutil.copy2(item, dest_item)

    log.info(f"Successfully promoted to {dest_path} (checkpoints excluded).")
    return dest_path
