"""Pytest fixtures and configuration."""

import os
import shutil
import tempfile

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure tests use an isolated temporary directory for SQLite and FAISS snapshots
@pytest.fixture(autouse=True)
def isolated_test_dir(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    temp_dir = tempfile.mkdtemp(prefix="cachesense_test_")
    db_path = os.path.join(temp_dir, "test_cache.db")
    faiss_dir = os.path.join(temp_dir, "test_faiss")
    
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setenv("FAISS_INDEX_PATH", faiss_dir)
    monkeypatch.setenv("MAX_CACHE_ENTRIES", "100")
    monkeypatch.setenv("DEFAULT_TTL_SECONDS", "3600")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.92")

    from app.config import settings
    settings.SQLITE_DB_PATH = db_path
    settings.FAISS_INDEX_PATH = faiss_dir
    settings.MAX_CACHE_ENTRIES = 100
    settings.DEFAULT_TTL_SECONDS = 3600
    settings.SIMILARITY_THRESHOLD = 0.92

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
