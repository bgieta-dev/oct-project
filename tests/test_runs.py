import logging
import os

from octseg.runs import (get_next_experiment_name, make_run_dir, promote_run_to_experiment,
                         send_discord_notification, setup_logging)


def test_make_run_dir_custom_and_default(tmp_path):
    custom_target = tmp_path / "custom_run"
    result = make_run_dir("runs", output=str(custom_target))
    assert result == custom_target
    assert custom_target.is_dir()

    default_dir = make_run_dir("evals", name="test_eval", output=str(tmp_path / "eval_out"))
    assert default_dir.is_dir()


def test_setup_logging(tmp_path):
    log_file = setup_logging(tmp_path, filename="test.log")
    assert os.path.exists(log_file)
    logger = logging.getLogger("test_logger")
    logger.info("Test log line")
    with open(log_file, "r") as f:
        content = f.read()
    assert "Test log line" in content


def test_send_discord_notification_without_webhook(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    # Must not raise or attempt network call
    send_discord_notification("Testing notification")


def test_get_next_experiment_name_current():
    next_name = get_next_experiment_name()
    assert next_name == "test19"


def test_get_next_experiment_name_custom(tmp_path):
    dummy_md = tmp_path / "EXPERIMENTS.md"
    dummy_md.write_text(
        "| dir | original run |\n| test5 | run1 |\n| test14 | run2 |\n"
    )
    assert get_next_experiment_name(dummy_md) == "test15"


def test_promote_run_to_experiment_excludes_checkpoints(tmp_path, monkeypatch):
    import octseg.runs
    monkeypatch.setattr(octseg.runs, "ROOT", tmp_path)

    run_dir = tmp_path / "outputs" / "runs" / "fake_run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("dummy_config: 1\n")
    (run_dir / "experiment.log").write_text("dummy_log\n")
    (run_dir / "best_model.pth").write_text("FAKE_PTH_WEIGHTS")
    (run_dir / "weights.pt").write_text("FAKE_PT_WEIGHTS")

    sub_dir = run_dir / "src"
    sub_dir.mkdir()
    (sub_dir / "test.py").write_text("print(1)\n")
    (sub_dir / "nested_ckpt.pth").write_text("NESTED_PTH")

    dest = promote_run_to_experiment(run_dir, "test99")
    assert dest.is_dir()
    assert (dest / "config.yaml").exists()
    assert (dest / "experiment.log").exists()
    assert (dest / "src" / "test.py").exists()
    # Checkpoints must NEVER be copied into experiments
    assert not (dest / "best_model.pth").exists()
    assert not (dest / "weights.pt").exists()
    assert not (dest / "src" / "nested_ckpt.pth").exists()
