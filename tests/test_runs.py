import logging
import os

from octseg.runs import make_run_dir, send_discord_notification, setup_logging


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
