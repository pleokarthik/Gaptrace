from click.testing import CliRunner

from gaptrace_capture.scaffold.cli import main


class TestNoSideEffects:
    def test_help_does_not_create_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert list(tmp_path.iterdir()) == []

    def test_bare_invocation_does_not_create_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 2
        assert "Usage:" in result.output
        assert list(tmp_path.iterdir()) == []


class TestInit:
    def test_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(main, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / "gaptrace_pipeline.py").exists()

    def test_second_call_fails_without_overwriting(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        CliRunner().invoke(main, ["init"])
        result = CliRunner().invoke(main, ["init"])
        assert result.exit_code == 1
