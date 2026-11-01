import os
import sys
from unittest.mock import patch

import pytest

from octseg.config import Config, save_config
import octseg.pipeline_hybrid as ph_mod
from octseg.pipeline_hybrid import main, pipeline_hybrid


def test_pipeline_hybrid_base_weights_missing(tmp_path):
    missing_base = str(tmp_path / "nonexistent_base.pth")
    out_dir = str(tmp_path / "out")
    cfg = Config()
    expert_cfg = Config()

    with pytest.raises(FileNotFoundError, match="Base model checkpoint not found"):
        pipeline_hybrid(
            base_weights=missing_base,
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
        )


def test_pipeline_hybrid_expert_weights_missing(tmp_path):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    missing_expert = str(tmp_path / "nonexistent_expert.pth")
    out_dir = str(tmp_path / "out")
    cfg = Config()
    expert_cfg = Config()

    with pytest.raises(FileNotFoundError, match="Expert model checkpoint not found"):
        pipeline_hybrid(
            base_weights=str(base_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            expert_weights=missing_expert,
        )


def test_pipeline_hybrid_train_expert_missing_checkpoint_raises(tmp_path):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    out_dir = str(tmp_path / "out")
    cfg = Config()
    expert_cfg = Config()

    with patch("octseg.pipeline_hybrid.train_expert") as mock_train:
        with pytest.raises(FileNotFoundError, match="Expert training completed but '.*' was not found"):
            pipeline_hybrid(
                base_weights=str(base_file),
                output_dir=out_dir,
                cfg=cfg,
                expert_cfg=expert_cfg,
                expert_weights=None,
            )
        mock_train.assert_called_once()


def test_pipeline_hybrid_fallback_when_sweep_best_none(tmp_path):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    expert_file = tmp_path / "expert.pth"
    expert_file.touch()
    out_dir = str(tmp_path / "out")
    cfg = Config()
    expert_cfg = Config()

    mock_metrics = {"mDice": 0.81, "class_dices": {1: 0.72}, "mHD95": 2.1}

    with patch("octseg.pipeline_hybrid.sweep_hybrid") as mock_sweep,          patch("octseg.pipeline_hybrid.evaluate_hybrid", return_value=mock_metrics) as mock_eval,          patch("octseg.pipeline_hybrid.archive_experiment_sources") as mock_archive:

        mock_sweep.return_value = {"best": None, "results": []}

        res = pipeline_hybrid(
            base_weights=str(base_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            expert_weights=str(expert_file),
        )

        assert res == mock_metrics
        mock_sweep.assert_called_once()
        mock_eval.assert_called_once_with(
            base_weights=str(base_file),
            expert_weights=str(expert_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            ensemble_mode=None,
            expert_weight=None,
            blend_strategy=None,
            irf_threshold=None,
            irf_min_region_size=None,
            irf_override=None,
        )
        mock_archive.assert_called_once_with(out_dir)


def test_pipeline_hybrid_fallback_when_sweep_best_has_no_params(tmp_path):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    expert_file = tmp_path / "expert.pth"
    expert_file.touch()
    out_dir = str(tmp_path / "out")
    cfg = Config()
    expert_cfg = Config()

    mock_metrics = {"mDice": 0.80, "class_dices": {"1": 0.70}, "mHD95": 2.2}

    with patch("octseg.pipeline_hybrid.sweep_hybrid") as mock_sweep,          patch("octseg.pipeline_hybrid.evaluate_hybrid", return_value=mock_metrics) as mock_eval,          patch("octseg.pipeline_hybrid.archive_experiment_sources"):

        mock_sweep.return_value = {"best": {"val_dice": 0.85}}

        res = pipeline_hybrid(
            base_weights=str(base_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            expert_weights=str(expert_file),
        )

        assert res == mock_metrics
        mock_eval.assert_called_once_with(
            base_weights=str(base_file),
            expert_weights=str(expert_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            ensemble_mode=None,
            expert_weight=None,
            blend_strategy=None,
            irf_threshold=None,
            irf_min_region_size=None,
            irf_override=None,
        )


def test_pipeline_hybrid_sweep_best_with_params(tmp_path):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    expert_file = tmp_path / "expert.pth"
    expert_file.touch()
    out_dir = str(tmp_path / "out")
    cfg = Config()
    expert_cfg = Config()

    best_params = {
        "ensemble_mode": "soft",
        "expert_weight": 0.65,
        "blend_strategy": "linear",
        "irf_threshold": 0.35,
        "irf_min_region_size": 25,
        "irf_override": True,
    }
    mock_metrics = {"mDice": 0.89, "class_dices": {1: 0.84}, "mHD95": 1.7}

    with patch("octseg.pipeline_hybrid.sweep_hybrid") as mock_sweep,          patch("octseg.pipeline_hybrid.evaluate_hybrid", return_value=mock_metrics) as mock_eval,          patch("octseg.pipeline_hybrid.archive_experiment_sources"):

        mock_sweep.return_value = {"best": {"params": best_params, "val_dice": 0.90}}

        res = pipeline_hybrid(
            base_weights=str(base_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            expert_weights=str(expert_file),
        )

        assert res == mock_metrics
        mock_eval.assert_called_once_with(
            base_weights=str(base_file),
            expert_weights=str(expert_file),
            output_dir=out_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            ensemble_mode="soft",
            expert_weight=0.65,
            blend_strategy="linear",
            irf_threshold=0.35,
            irf_min_region_size=25,
            irf_override=True,
        )


def test_pipeline_hybrid_train_and_promote(tmp_path):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    out_dir = tmp_path / "out"
    cfg = Config()
    expert_cfg = Config()

    def fake_train_expert(expert_run_dir, expert_cfg_arg, epochs=None):
        ckpt = os.path.join(expert_run_dir, "best_model.pth")
        with open(ckpt, "w") as f:
            f.write("CHECKPOINT")

    def fake_sweep(base_weights, expert_weights, output_dir, cfg, expert_cfg, split):
        os.makedirs(output_dir, exist_ok=True)
        sweep_yaml = os.path.join(output_dir, "sweep.yaml")
        with open(sweep_yaml, "w") as f:
            f.write("best: null\n")
        return {"best": None}

    mock_metrics = {"mDice": 0.85, "class_dices": {1: 0.75}, "mHD95": 2.0}

    with patch("octseg.pipeline_hybrid.train_expert", side_effect=fake_train_expert) as mock_train,          patch("octseg.pipeline_hybrid.sweep_hybrid", side_effect=fake_sweep),          patch("octseg.pipeline_hybrid.evaluate_hybrid", return_value=mock_metrics),          patch("octseg.pipeline_hybrid.archive_experiment_sources") as mock_archive,          patch("octseg.pipeline_hybrid.promote_run_to_experiment") as mock_promote:

        res = pipeline_hybrid(
            base_weights=str(base_file),
            output_dir=str(out_dir),
            cfg=cfg,
            expert_cfg=expert_cfg,
            expert_weights=None,
            epochs=7,
            promote=True,
            exp_name="test_exp_42",
        )

        assert res == mock_metrics
        mock_train.assert_called_once_with(os.path.join(str(out_dir), "expert"), expert_cfg, epochs=7)
        mock_archive.assert_called_once_with(str(out_dir))
        mock_promote.assert_called_once_with(str(out_dir), "test_exp_42")
        assert (out_dir / "sweep.yaml").exists()


def test_main_cli_argument_parsing(tmp_path, monkeypatch):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    expert_file = tmp_path / "expert.pth"
    expert_file.touch()
    cfg_file = tmp_path / "cfg.yaml"
    exp_cfg_file = tmp_path / "expert_cfg.yaml"
    save_config(Config(), str(cfg_file))
    save_config(Config(), str(exp_cfg_file))

    out_dir = tmp_path / "custom_out"
    test_metrics = {"mDice": 0.8765, "class_dices": {1: 0.8123}, "mHD95": 2.34}

    test_args = [
        "pipeline_hybrid.py",
        "--base-weights", str(base_file),
        "--expert-weights", str(expert_file),
        "--config", str(cfg_file),
        "--expert-config", str(exp_cfg_file),
        "--name", "my_custom_run",
        "--output", str(out_dir),
        "--epochs", "12",
        "--promote",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("octseg.pipeline_hybrid.pipeline_hybrid", return_value=test_metrics) as mock_pipeline,          patch("octseg.pipeline_hybrid.send_discord_notification") as mock_discord:

        main()

        mock_pipeline.assert_called_once_with(
            base_weights=str(base_file),
            output_dir=str(out_dir),
            cfg=Config(),
            expert_cfg=Config(),
            expert_weights=str(expert_file),
            epochs=12,
            promote=True,
            exp_name="my_custom_run",
        )
        mock_discord.assert_called_once()
        assert "mDice: `0.8765`" in mock_discord.call_args[0][0]
        assert "IRF Dice: `0.8123`" in mock_discord.call_args[0][0]


def test_main_promote_default_name_resolves_next_exp(tmp_path, monkeypatch):
    base_file = tmp_path / "base.pth"
    base_file.touch()
    cfg_file = tmp_path / "cfg.yaml"
    exp_cfg_file = tmp_path / "expert_cfg.yaml"
    save_config(Config(), str(cfg_file))
    save_config(Config(), str(exp_cfg_file))

    out_dir = tmp_path / "default_out"
    test_metrics = {"mDice": 0.75, "class_dices": {"1": 0.65}, "mHD95": 3.5}

    test_args = [
        "pipeline_hybrid.py",
        "--base-weights", str(base_file),
        "--config", str(cfg_file),
        "--expert-config", str(exp_cfg_file),
        "--output", str(out_dir),
        "--promote",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("octseg.pipeline_hybrid.get_next_experiment_name", return_value="test88") as mock_next_exp,          patch("octseg.pipeline_hybrid.pipeline_hybrid", return_value=test_metrics) as mock_pipeline,          patch("octseg.pipeline_hybrid.send_discord_notification"):

        main()

        mock_next_exp.assert_called_once()
        mock_pipeline.assert_called_once()
        assert mock_pipeline.call_args[1]["exp_name"] == "test88"


def test_main_resolves_relative_to_root(tmp_path, monkeypatch):
    rel_base = tmp_path / "relative_base.pth"
    rel_base.touch()
    rel_expert = tmp_path / "relative_expert.pth"
    rel_expert.touch()

    cfg_file = tmp_path / "cfg.yaml"
    exp_cfg_file = tmp_path / "expert_cfg.yaml"
    save_config(Config(), str(cfg_file))
    save_config(Config(), str(exp_cfg_file))

    monkeypatch.setattr(ph_mod, "ROOT", tmp_path)

    test_args = [
        "pipeline_hybrid.py",
        "--base-weights", "relative_base.pth",
        "--expert-weights", "relative_expert.pth",
        "--config", str(cfg_file),
        "--expert-config", str(exp_cfg_file),
        "--output", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    test_metrics = {"mDice": 0.8, "class_dices": {}, "mHD95": 2.0}

    with patch("octseg.pipeline_hybrid.pipeline_hybrid", return_value=test_metrics) as mock_pipeline,          patch("octseg.pipeline_hybrid.send_discord_notification"):

        main()

        assert mock_pipeline.call_args[1]["base_weights"] == str(rel_base)
        assert mock_pipeline.call_args[1]["expert_weights"] == str(rel_expert)


def test_main_handles_exception_and_exits(tmp_path, monkeypatch):
    test_args = [
        "pipeline_hybrid.py",
        "--base-weights", str(tmp_path / "missing.pth"),
        "--config", str(tmp_path / "cfg.yaml"),
        "--expert-config", str(tmp_path / "expert_cfg.yaml"),
        "--output", str(tmp_path / "out"),
    ]
    save_config(Config(), str(tmp_path / "cfg.yaml"))
    save_config(Config(), str(tmp_path / "expert_cfg.yaml"))

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("octseg.pipeline_hybrid.pipeline_hybrid", side_effect=FileNotFoundError("Base model checkpoint not found")),          patch("octseg.pipeline_hybrid.send_discord_notification") as mock_discord:

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_discord.assert_called_once()
        assert "CRITICAL FAILURE" in mock_discord.call_args[0][0]
