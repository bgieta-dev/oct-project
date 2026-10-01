import argparse
import logging

from octseg.config import CONFIG_DIR, Config, load_config, save_config
from octseg.runs import archive_experiment_sources, make_run_dir, setup_logging
from octseg.train import train_model

log = logging.getLogger(__name__)


def train_expert(run_dir, cfg: Config, epochs=None):
    """
    Trains the Binary IRF Expert using the modular train_model logic.
    Target: mit-b0 architecture with high-recall Tversky loss.
    Best weights are written to ``<run_dir>/best_model.pth``.
    """
    log.info("--- STARTING IRF EXPERT TRAINING ---")
    log.info(f"Target Class: {cfg.CLASS_NAMES[cfg.TARGET_CLASS]}")
    log.info(f"Model: {cfg.MODEL_NAME} | Tversky Beta: {cfg.TVERSKY_BETA} | 2.5D: {cfg.USE_25D}")
    train_model(str(run_dir), cfg=cfg, epochs=epochs)
    log.info(f"Expert training finished. Best weights saved to {run_dir}/best_model.pth")


def main():
    parser = argparse.ArgumentParser(description="Train the binary IRF expert")
    parser.add_argument("--config", default=str(CONFIG_DIR / "irf_expert.yaml"), help="YAML config (default: configs/irf_expert.yaml)")
    parser.add_argument("--output", default=None, help="Run directory (default: outputs/runs/<timestamp>_<name>)")
    parser.add_argument("--name", default="irf_expert", help="Run name suffix (default: irf_expert)")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = make_run_dir("runs", args.name, args.output)
    setup_logging(run_dir)
    save_config(cfg, run_dir / "config.yaml")
    log.info(f"Results directory: {run_dir}")
    train_expert(run_dir, cfg, epochs=args.epochs)
    archive_experiment_sources(str(run_dir))


if __name__ == "__main__":
    main()
