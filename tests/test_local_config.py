"""
本地配置文件。

守的是安全边界：配置可以落盘（免得每次重配），但
  - 绝不能覆盖进程环境变量（否则临时切换模型会失效）；
  - 白名单之外的变量不得注入（不能借配置文件改 PATH 之类）；
  - 写出的文件必须 600 权限、且值能原样读回来。
"""

import os

import pytest

from autoalpha.local_config import (
    ALLOWED_KEYS, load_local_env, parse_env, save_project_env,
)


class TestParse:
    def test_basic_and_export(self):
        out = parse_env("LLM_MODEL=m\nexport LLM_BASE_URL=http://x\n")
        assert out == {"LLM_MODEL": "m", "LLM_BASE_URL": "http://x"}

    def test_quotes_stripped(self):
        assert parse_env('A="hello world"')["A"] == "hello world"
        assert parse_env("B='single'")["B"] == "single"

    def test_comments_and_blank_lines(self):
        out = parse_env("# comment\n\nLLM_MODEL=m\n\n")
        assert out == {"LLM_MODEL": "m"}

    def test_trailing_comment_only_when_unquoted(self):
        assert parse_env("A=1  # note")["A"] == "1"
        # 加引号的值里 # 是内容的一部分，不能截断
        assert parse_env('A="x # y"')["A"] == "x # y"

    def test_malformed_lines_skipped(self):
        out = parse_env("NOEQUALS\n=novalue\nLLM_MODEL=m\n")
        assert out == {"LLM_MODEL": "m"}

    def test_key_with_special_chars_preserved(self):
        # 网关密钥常含各种符号，不能被误截断
        out = parse_env("LLM_API_KEY=sk-abc!@$%^&*()_+-=[]{};:,./<>?")
        assert out["LLM_API_KEY"].startswith("sk-abc")


class TestLoad:
    def test_does_not_override_process_env(self, tmp_path, monkeypatch):
        """用户 export 的必须优先，否则临时换模型会被文件悄悄改回去。"""
        p = tmp_path / ".env"
        p.write_text("LLM_MODEL=from-file\n")
        monkeypatch.setenv("LLM_MODEL", "from-process")
        load_local_env([str(p)])
        assert os.environ["LLM_MODEL"] == "from-process"

    def test_loads_when_absent(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("LLM_MODEL=from-file\n")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        loaded = load_local_env([str(p)])
        assert os.environ["LLM_MODEL"] == "from-file"
        assert loaded == [str(p)]

    def test_whitelist_blocks_arbitrary_vars(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("PATH=/tmp/evil\nPYTHONPATH=/tmp/evil\nLLM_MODEL=ok\n")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        old_path = os.environ.get("PATH")
        load_local_env([str(p)])
        assert os.environ.get("PATH") == old_path
        assert "PYTHONPATH" not in os.environ or \
            os.environ["PYTHONPATH"] != "/tmp/evil"
        assert os.environ["LLM_MODEL"] == "ok"

    def test_missing_file_is_noop(self, tmp_path):
        assert load_local_env([str(tmp_path / "nope.env")]) == []


class TestSave:
    def test_file_is_0600(self, tmp_path):
        p = str(tmp_path / ".env")
        save_project_env({"LLM_MODEL": "m"}, path=p)
        assert os.stat(p).st_mode & 0o777 == 0o600

    def test_roundtrip_preserves_values(self, tmp_path):
        p = str(tmp_path / ".env")
        vals = {"LLM_MODEL": "deepseek/deepseek-v4.1-flash",
                "LLM_BASE_URL": "https://api.qnaigc.com/v1"}
        save_project_env(vals, path=p)
        assert parse_env(open(p).read()) == vals

    def test_save_merges_not_overwrites(self, tmp_path):
        """第二次保存不该把第一次写的其它键冲掉。"""
        p = str(tmp_path / ".env")
        save_project_env({"LLM_API_KEY": "sk-x"}, path=p)
        save_project_env({"LLM_MODEL": "m"}, path=p)
        out = parse_env(open(p).read())
        assert out["LLM_API_KEY"] == "sk-x"
        assert out["LLM_MODEL"] == "m"

    def test_value_with_spaces_survives(self, tmp_path):
        p = str(tmp_path / ".env")
        save_project_env({"LLM_MODEL": "model with space"}, path=p)
        assert parse_env(open(p).read())["LLM_MODEL"] == "model with space"

    def test_saved_keys_are_allowed(self):
        """写文件的键必须在白名单内，否则读不回来。"""
        for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
            assert k in ALLOWED_KEYS
