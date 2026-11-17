"""Tests for the --real flag of scripts/run_assessment.py.

No test calls a real model: the model loader is replaced by a fake.
"""

from scripts import run_assessment


def test_default_is_fake_mode(tmp_path, capsys):
    code = run_assessment.cli([str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "FAKE agents only" in out
    assert len(list(tmp_path.iterdir())) == 1


def test_real_stops_on_missing_project_before_loading_model(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_assessment, "_load_model", lambda: calls.append("loaded"))
    code = run_assessment.cli(
        ["--real", "--project", str(tmp_path / "nope"), str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "Project folder not found" in out
    assert calls == []


def test_real_stops_clearly_when_key_is_missing(tmp_path, monkeypatch, capsys):
    def no_key():
        raise ValueError("GROQ_API_KEY is missing. Check your local .env file.")

    monkeypatch.setattr(run_assessment, "_load_model", no_key)
    code = run_assessment.cli(["--real", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "REAL model" in out
    assert "GROQ_API_KEY is missing" in out
    assert list(tmp_path.iterdir()) == []