"""
Tests for issue #203: token refresh race with a concurrent refresher sharing the
same credential store (host kiro-cli, Kiro IDE, or a second gateway container).

Covers:
- stale in-memory refresh token + newer on-disk token -> adopt on-disk, do not refresh
- SQLite write-back compare-and-set aborts instead of clobbering a rotated token
- a normal successful refresh still persists
- JSON credential write is atomic (temp + os.replace) and never loses the original
- no token/secret values are logged
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from kiro.auth import KiroAuthManager


def _token_row(access_token, refresh_token, expires_at):
    return json.dumps({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at.isoformat(),
        "region": "us-east-1",
        "startUrl": "https://example.awsapps.com/start",
    })


def _make_db(path, access_token, refresh_token, expires_at):
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value TEXT)")
    cur.execute(
        "INSERT INTO auth_kv (key, value) VALUES (?, ?)",
        ("kirocli:odic:token", _token_row(access_token, refresh_token, expires_at)),
    )
    cur.execute(
        "INSERT INTO auth_kv (key, value) VALUES (?, ?)",
        ("kirocli:odic:device-registration",
         json.dumps({"client_id": "cid", "client_secret": "csecret", "region": "us-east-1"})),
    )
    conn.commit()
    conn.close()


def _read_row(path, key="kirocli:odic:token"):
    conn = sqlite3.connect(str(path))
    row = conn.execute("SELECT value FROM auth_kv WHERE key = ?", (key,)).fetchone()
    conn.close()
    return json.loads(row[0])


class TestAdoptOnDiskCredentials:
    """The gateway must not spend a refresh token another process already rotated."""

    @pytest.mark.asyncio
    async def test_stale_in_memory_token_adopts_on_disk_and_skips_refresh(self, tmp_path, monkeypatch):
        db = tmp_path / "data.sqlite3"
        future = datetime.now(timezone.utc) + timedelta(hours=8)
        _make_db(db, "old_access", "old_refresh", datetime.now(timezone.utc) - timedelta(minutes=1))

        manager = KiroAuthManager(sqlite_db=str(db))
        assert manager._refresh_token == "old_refresh"

        # Simulate host kiro-cli refreshing and persisting a brand new credential
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE auth_kv SET value = ? WHERE key = ?",
            (_token_row("cli_access", "cli_refresh", future), "kirocli:odic:token"),
        )
        conn.commit()
        conn.close()

        called = []

        async def _must_not_refresh():
            called.append(True)

        monkeypatch.setattr(manager, "_do_aws_sso_oidc_refresh", _must_not_refresh)
        monkeypatch.setattr(manager, "_refresh_token_kiro_desktop", _must_not_refresh)

        token = await manager.get_access_token()

        assert token == "cli_access"
        assert manager._refresh_token == "cli_refresh"
        assert called == [], "must not refresh a token that was already rotated on disk"

    @pytest.mark.asyncio
    async def test_adopted_token_is_used_for_refresh_when_also_stale(self, tmp_path, monkeypatch):
        db = tmp_path / "data.sqlite3"
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        _make_db(db, "old_access", "old_refresh", past)

        manager = KiroAuthManager(sqlite_db=str(db))

        # On-disk token rotated but also already expired -> refresh must use the new one
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE auth_kv SET value = ? WHERE key = ?",
            (_token_row("cli_access", "cli_refresh", past), "kirocli:odic:token"),
        )
        conn.commit()
        conn.close()

        used = {}

        async def _fake_refresh():
            used["refresh_token"] = manager._refresh_token
            manager._access_token = "fresh_access"
            manager._refresh_token = "fresh_refresh"
            manager._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        monkeypatch.setattr(manager, "_do_aws_sso_oidc_refresh", _fake_refresh)
        monkeypatch.setattr(manager, "_refresh_token_kiro_desktop", _fake_refresh)

        token = await manager.get_access_token()

        assert token == "fresh_access"
        assert used["refresh_token"] == "cli_refresh", "must refresh with the adopted on-disk token"

    @pytest.mark.asyncio
    async def test_force_refresh_reloads_before_refreshing(self, tmp_path, monkeypatch):
        """The 403 -> force_refresh path must not burn a stale token repeatedly."""
        db = tmp_path / "data.sqlite3"
        _make_db(db, "old_access", "old_refresh", datetime.now(timezone.utc) - timedelta(minutes=1))
        manager = KiroAuthManager(sqlite_db=str(db))

        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE auth_kv SET value = ? WHERE key = ?",
            (_token_row("cli_access", "cli_refresh", datetime.now(timezone.utc) + timedelta(hours=8)),
             "kirocli:odic:token"),
        )
        conn.commit()
        conn.close()

        calls = []

        async def _count_refresh():
            calls.append(True)

        monkeypatch.setattr(manager, "_do_aws_sso_oidc_refresh", _count_refresh)
        monkeypatch.setattr(manager, "_refresh_token_kiro_desktop", _count_refresh)

        token = await manager.force_refresh()

        assert token == "cli_access"
        assert calls == []


class TestSqliteCompareAndSet:
    """Write-back must be an atomic compare-and-set, not last-writer-wins."""

    def test_cas_aborts_when_stored_token_changed_underneath(self, tmp_path):
        db = tmp_path / "data.sqlite3"
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        _make_db(db, "old_access", "old_refresh", past)

        manager = KiroAuthManager(sqlite_db=str(db))
        assert manager._loaded_refresh_token == "old_refresh"

        # Concurrent refresher rotates the credential after we loaded it
        winner_expiry = datetime.now(timezone.utc) + timedelta(hours=8)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "UPDATE auth_kv SET value = ? WHERE key = ?",
            (_token_row("winner_access", "winner_refresh", winner_expiry), "kirocli:odic:token"),
        )
        conn.commit()
        conn.close()

        # Our (now stale) refresh result tries to persist
        manager._access_token = "loser_access"
        manager._refresh_token = "loser_refresh"
        manager._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        manager._save_credentials_to_sqlite()

        stored = _read_row(db)
        assert stored["refresh_token"] == "winner_refresh", "must not clobber the newer token"
        assert stored["access_token"] == "winner_access"
        assert manager._last_write_conflict is True
        # And it reloaded rather than keeping a dead credential in memory
        assert manager._refresh_token == "winner_refresh"
        assert manager._access_token == "winner_access"

    def test_successful_save_still_persists_and_preserves_unknown_fields(self, tmp_path):
        db = tmp_path / "data.sqlite3"
        _make_db(db, "old_access", "old_refresh", datetime.now(timezone.utc) - timedelta(minutes=1))

        manager = KiroAuthManager(sqlite_db=str(db))
        manager._access_token = "new_access"
        manager._refresh_token = "new_refresh"
        manager._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        manager._save_credentials_to_sqlite()

        stored = _read_row(db)
        assert stored["access_token"] == "new_access"
        assert stored["refresh_token"] == "new_refresh"
        assert stored["startUrl"] == "https://example.awsapps.com/start"
        assert manager._last_write_conflict is False
        # Baseline advanced, so a second save is not seen as a conflict
        manager._access_token = "newer_access"
        manager._refresh_token = "newer_refresh"
        manager._save_credentials_to_sqlite()
        assert _read_row(db)["refresh_token"] == "newer_refresh"

    def test_write_uses_immediate_transaction(self, tmp_path, monkeypatch):
        db = tmp_path / "data.sqlite3"
        _make_db(db, "a", "old_refresh", datetime.now(timezone.utc) - timedelta(minutes=1))
        manager = KiroAuthManager(sqlite_db=str(db))

        statements = []
        real_connect = sqlite3.connect

        def _tracing_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            real_execute = conn.execute

            class _Cur:
                def __init__(self, inner):
                    self._inner = inner

                def execute(self, sql, *a, **kw):
                    statements.append(sql.strip().upper())
                    return self._inner.execute(sql, *a, **kw)

                def __getattr__(self, name):
                    return getattr(self._inner, name)

            class _Conn:
                def cursor(self):
                    return _Cur(conn.cursor())

                def execute(self, sql, *a, **kw):
                    statements.append(sql.strip().upper())
                    return real_execute(sql, *a, **kw)

                def __getattr__(self, name):
                    return getattr(conn, name)

            return _Conn()

        monkeypatch.setattr(sqlite3, "connect", _tracing_connect)
        manager._access_token = "x"
        manager._refresh_token = "y"
        manager._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        manager._save_credentials_to_sqlite()

        assert any(s.startswith("BEGIN IMMEDIATE") for s in statements)
        assert any("AND VALUE = ?" in s for s in statements)


class TestAtomicJsonWrite:
    """The JSON (Enterprise / Kiro Desktop) credential write must be atomic."""

    def _creds(self, tmp_path):
        creds = tmp_path / "credentials.json"
        creds.write_text(json.dumps({
            "accessToken": "old_access",
            "refreshToken": "old_refresh",
            "region": "us-east-1",
            "customField": "keep-me",
        }), encoding="utf-8")
        return creds

    def test_uses_temp_file_and_replace(self, tmp_path, monkeypatch):
        creds = self._creds(tmp_path)
        manager = KiroAuthManager(creds_file=str(creds))

        replaced = {}
        real_replace = os.replace

        def _spy(src, dst, *a, **kw):
            replaced["src"] = str(src)
            replaced["dst"] = str(dst)
            # The temp file must live in the same directory as the target
            assert os.path.dirname(str(src)) == os.path.dirname(str(dst))
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(os, "replace", _spy)

        manager._access_token = "new_access"
        manager._refresh_token = "new_refresh"
        manager._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        manager._save_credentials_to_file()

        assert replaced["dst"] == str(creds)
        assert replaced["src"] != str(creds)
        data = json.loads(creds.read_text(encoding="utf-8"))
        assert data["refreshToken"] == "new_refresh"
        assert data["customField"] == "keep-me"
        # No leftover temp files
        assert sorted(p.name for p in tmp_path.iterdir()) == ["credentials.json"]

    def test_original_intact_and_no_temp_left_on_failure(self, tmp_path, monkeypatch):
        creds = self._creds(tmp_path)
        manager = KiroAuthManager(creds_file=str(creds))

        def _boom(src, dst, *a, **kw):
            raise OSError("simulated failure")

        monkeypatch.setattr(os, "replace", _boom)

        manager._access_token = "new_access"
        manager._refresh_token = "new_refresh"
        manager._save_credentials_to_file()  # must not raise

        data = json.loads(creds.read_text(encoding="utf-8"))
        assert data["refreshToken"] == "old_refresh", "original file must survive a failed write"
        assert data["accessToken"] == "old_access"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["credentials.json"]

    def test_json_path_adopts_rotated_on_disk_token(self, tmp_path, monkeypatch):
        creds = self._creds(tmp_path)
        manager = KiroAuthManager(creds_file=str(creds))
        assert manager._refresh_token == "old_refresh"

        creds.write_text(json.dumps({
            "accessToken": "ide_access",
            "refreshToken": "ide_refresh",
            "region": "us-east-1",
            "expiresAt": (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(),
        }), encoding="utf-8")

        assert manager._adopt_on_disk_credentials_if_changed() is True
        assert manager._refresh_token == "ide_refresh"
        assert manager._access_token == "ide_access"

    def test_no_adoption_when_disk_matches_memory(self, tmp_path):
        creds = self._creds(tmp_path)
        manager = KiroAuthManager(creds_file=str(creds))
        assert manager._adopt_on_disk_credentials_if_changed() is False


class TestNoSecretLeakage:
    """Token and secret values must never reach the logs."""

    def test_conflict_and_adoption_paths_log_no_secret_values(self, tmp_path):
        from loguru import logger

        records = []
        sink_id = logger.add(lambda m: records.append(m), level="DEBUG")
        try:
            db = tmp_path / "data.sqlite3"
            _make_db(db, "SEKRET_ACCESS", "SEKRET_REFRESH",
                     datetime.now(timezone.utc) - timedelta(minutes=1))
            manager = KiroAuthManager(sqlite_db=str(db))

            conn = sqlite3.connect(str(db))
            conn.execute(
                "UPDATE auth_kv SET value = ? WHERE key = ?",
                (_token_row("OTHER_ACCESS", "OTHER_REFRESH",
                            datetime.now(timezone.utc) + timedelta(hours=8)),
                 "kirocli:odic:token"),
            )
            conn.commit()
            conn.close()

            manager._adopt_on_disk_credentials_if_changed()
            manager._access_token = "LOSER_ACCESS"
            manager._refresh_token = "LOSER_REFRESH"
            manager._loaded_refresh_token = "SEKRET_REFRESH"
            manager._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            manager._save_credentials_to_sqlite()
        finally:
            logger.remove(sink_id)

        blob = "".join(records)
        for secret in ("SEKRET_ACCESS", "SEKRET_REFRESH", "OTHER_ACCESS",
                       "OTHER_REFRESH", "LOSER_ACCESS", "LOSER_REFRESH", "csecret"):
            assert secret not in blob, f"secret value {secret!r} leaked into logs"
