from intraday_engine.versioning import (
    EXECUTION_MODEL_VERSION,
    FEATURE_ENGINE_VERSION,
    STRATEGY_VERSION,
    get_code_commit,
)


def test_version_constants_are_non_empty_strings():
    for version in (STRATEGY_VERSION, FEATURE_ENGINE_VERSION, EXECUTION_MODEL_VERSION):
        assert isinstance(version, str) and version


def test_get_code_commit_never_raises_and_returns_str_or_none():
    result = get_code_commit()
    assert result is None or isinstance(result, str)


def test_get_code_commit_returns_none_when_git_is_unavailable(monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert get_code_commit() is None
