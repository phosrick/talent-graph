# -*- coding: utf-8 -*-
"""Unit tests for configuration center"""

import os
from pathlib import Path
from configs.settings import Settings, get_settings


def test_default_settings():
    # 测试未被 .env 覆盖时的原生默认配置
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "deepseek"
    assert settings.deepseek_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_max_concurrency == 20
    assert settings.mock_email is True
    assert settings.enable_telemetry is False
    assert isinstance(settings.cors_origins, list)


def test_cors_origins_parsing():
    s = Settings(cors_origins="http://localhost:3000, http://localhost:8000")
    assert s.cors_origins == ["http://localhost:3000", "http://localhost:8000"]

    s_star = Settings(cors_origins="*")
    assert s_star.cors_origins == ["*"]


def test_directory_creation(tmp_path: Path):
    cache_dir = tmp_path / "test_cache"
    projects_dir = tmp_path / "test_projects"
    s = Settings(corpus_cache_dir=cache_dir, projects_dir=projects_dir)
    s.ensure_directories()
    assert cache_dir.exists()
    assert projects_dir.exists()
    assert s.get_project_dir("p1").name == "p1"
    assert s.get_project_report_file("p1").name == "report.json"
    assert s.get_project_graphs_dir("p1").name == "graphs"
    assert s.get_project_pdf_dir("p1").name == "pdf_reports"
