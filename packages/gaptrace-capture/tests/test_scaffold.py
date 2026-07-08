from gaptrace_capture.scaffold.template import generate_scaffold


class TestGenerateScaffold:
    def test_creates_file(self, tmp_path):
        path = generate_scaffold(tmp_path)
        assert path.exists()
        assert path.name == "ctx_pipeline.py"
        content = path.read_text(encoding="utf-8")
        assert "gaptrace_capture.start" in content
        old_alias = "gaptrace" + "run"  # retired import name; concat keeps repo grep clean
        assert old_alias not in content
        assert "Stage 1: Retrieval" in content
        assert "Stage 4: LLM call" in content

    def test_raises_on_second_call(self, tmp_path):
        generate_scaffold(tmp_path)
        try:
            generate_scaffold(tmp_path)
            assert False, "Should have raised FileExistsError"
        except FileExistsError as e:
            assert "already exists" in str(e)

    def test_default_path_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = generate_scaffold()
        assert path.parent == tmp_path
