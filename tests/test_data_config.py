"""
数据源配置与选路测试。
"""

import pytest

from autoalpha.data_config import (
    DataSourceRegistry, DataSourceConfig, PROVIDERS, module_available,
)


def test_defaults_to_local_when_nothing_configured():
    """不配置任何账号就走免费平台，不能是报错或空。"""
    assert DataSourceRegistry().active() == "local"


def test_configured_paid_source_wins_when_ready(monkeypatch):
    """配置了账号则优先走用户配置的平台。"""
    import autoalpha.data_config as dc
    monkeypatch.setattr(dc, "module_available", lambda n: True)
    r = DataSourceRegistry()
    r.set("tushare_pro", credentials={"token": "t"})
    assert r.active() == "tushare_pro", "付费源就绪时应优先于 local"


def test_missing_library_falls_back_to_free(monkeypatch):
    """
    依赖库没装时不能静默失败，也不能让流水线挂掉 —— 回落到免费源。
    本环境实测 akshare/tushare 等全部未安装，所以这是常态而非边界。
    """
    import autoalpha.data_config as dc
    monkeypatch.setattr(dc, "module_available",
                        lambda n: n == "local")
    r = DataSourceRegistry()
    r.set("tushare_pro", credentials={"token": "t"})
    assert r.active() == "local"


def test_missing_credentials_blocks_readiness(monkeypatch):
    import autoalpha.data_config as dc
    monkeypatch.setattr(dc, "module_available", lambda n: True)
    r = DataSourceRegistry()
    r.set("joinquant", credentials={"username": "u"})      # 缺 password
    assert r.active() == "local"
    cfg = DataSourceConfig("joinquant", credentials={"username": "u"})
    assert cfg.missing_credentials() == ["password"]


def test_blank_credential_keeps_previous_value():
    """留空表示不改，与 LLM 配置同一约定。"""
    r = DataSourceRegistry()
    r.set("tushare_pro", credentials={"token": "original"})
    r.set("tushare_pro", credentials={"token": ""})
    assert r._configs["tushare_pro"].credentials["token"] == "original"


def test_disabled_source_is_not_used(monkeypatch):
    import autoalpha.data_config as dc
    monkeypatch.setattr(dc, "module_available", lambda n: True)
    r = DataSourceRegistry()
    r.set("tushare_pro", credentials={"token": "t"})
    assert r.active() == "tushare_pro"
    r.set("tushare_pro", enabled=False)
    assert r.active() == "local"


def test_status_never_exposes_credentials():
    """状态接口绝不能回传凭据值。"""
    r = DataSourceRegistry()
    r.set("tushare_pro", credentials={"token": "SECRET-TOKEN-XYZ"})
    blob = repr(r.status())
    assert "SECRET-TOKEN-XYZ" not in blob


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        DataSourceRegistry().set("not_a_real_source")


def test_local_cannot_be_removed():
    r = DataSourceRegistry()
    r.remove("local")
    assert r.active() == "local"


def test_every_provider_declares_required_metadata():
    for name, meta in PROVIDERS.items():
        assert meta["label"] and meta["note"]
        assert isinstance(meta["credentials"], list)
        assert isinstance(meta["free"], bool)
