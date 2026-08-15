
"""
Tests for account initialization diagnostics (FIX-07).

Tests cover:
- Per-account failure reason is captured and logged with the concrete cause
- Aggregated RuntimeError from lifespan() contains the per-account reasons
- OIDC invalid_grant produces re-authentication guidance
- Secrets (tokens, client secrets, Authorization headers) never leak
- Successful initialization is unaffected
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from loguru import logger

from kiro.account_manager import (
    AccountManager,
    MAX_ERROR_BODY_CHARS,
    describe_init_failure,
    format_init_failure_guidance,
    is_invalid_grant,
    redact_secrets,
)


INVALID_GRANT_BODY = '{"error":"invalid_grant","error_description":"Resource not found"}'


class _FakeResponse:
    """Minimal stand-in for httpx.Response."""
    
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


class _FakeHTTPStatusError(Exception):
    """Minimal stand-in for httpx.HTTPStatusError (carries .response)."""
    
    def __init__(self, message: str, response: _FakeResponse):
        super().__init__(message)
        self.response = response


@pytest.fixture
def log_sink():
    """Capture loguru output for assertions."""
    records = []
    sink_id = logger.add(lambda msg: records.append(msg), level="DEBUG")
    yield records
    logger.remove(sink_id)


def _make_manager(tmp_path, cred_path):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps([
        {"type": "json", "path": str(cred_path), "enabled": True}
    ]))
    return AccountManager(
        credentials_file=str(creds_file),
        state_file=str(tmp_path / "state.json")
    )


# =============================================================================
# Test Class: Reason formatting helpers
# =============================================================================

class TestFailureReasonHelpers:
    """Tests for describe_init_failure / redact_secrets / guidance helpers."""
    
    def test_describe_plain_exception(self):
        """Reason includes exception type and message."""
        reason = describe_init_failure(ValueError("bad region"))
        assert "ValueError" in reason
        assert "bad region" in reason
    
    def test_describe_http_error_includes_status_and_body(self):
        """HTTP errors carry the upstream status and error body."""
        exc = _FakeHTTPStatusError(
            "400 Bad Request",
            _FakeResponse(400, INVALID_GRANT_BODY)
        )
        reason = describe_init_failure(exc)
        
        assert "HTTP 400" in reason
        assert "invalid_grant" in reason
        assert "Resource not found" in reason
    
    def test_describe_truncates_long_body(self):
        """Oversized upstream bodies are truncated."""
        exc = _FakeHTTPStatusError("boom", _FakeResponse(500, "x" * 5000))
        reason = describe_init_failure(exc)
        
        assert len(reason) <= MAX_ERROR_BODY_CHARS + len("... (truncated)")
        assert reason.endswith("... (truncated)")
    
    def test_redact_secrets_json_fields(self):
        """Token-like JSON fields are redacted."""
        text = ('{"accessToken": "AT-SECRET", "refreshToken": "RT-SECRET", '
                '"clientSecret": "CS-SECRET"}')
        redacted = redact_secrets(text)
        
        assert "AT-SECRET" not in redacted
        assert "RT-SECRET" not in redacted
        assert "CS-SECRET" not in redacted
        assert "[REDACTED]" in redacted
    
    def test_redact_secrets_authorization_header(self):
        """Authorization headers are redacted, prefix included."""
        redacted = redact_secrets("Authorization: Bearer HEADER-SECRET")
        assert "HEADER-SECRET" not in redacted
    
    def test_invalid_grant_detection(self):
        """invalid_grant is recognised regardless of case."""
        assert is_invalid_grant(describe_init_failure(
            _FakeHTTPStatusError("x", _FakeResponse(400, INVALID_GRANT_BODY))
        )) is True
        assert is_invalid_grant("ValueError: bad region") is False
        assert is_invalid_grant(None) is False
    
    def test_guidance_for_invalid_grant(self):
        """invalid_grant gets explicit re-authentication guidance."""
        reason = describe_init_failure(
            _FakeHTTPStatusError("x", _FakeResponse(400, INVALID_GRANT_BODY))
        )
        guidance = format_init_failure_guidance(reason)
        
        assert "re-authenticated" in guidance
        assert "Kiro IDE" in guidance
        assert "kiro-cli" in guidance
        assert "not a configuration error" in guidance
    
    def test_guidance_passthrough_for_other_errors(self):
        """Non-invalid_grant reasons are returned unchanged."""
        assert format_init_failure_guidance("ValueError: nope") == "ValueError: nope"
        assert format_init_failure_guidance(None) == "unknown error"


# =============================================================================
# Test Class: AccountManager records failure reasons
# =============================================================================

class TestInitializeAccountRecordsReason:
    """Tests that _initialize_account records and logs the concrete cause."""
    
    @pytest.mark.asyncio
    async def test_failure_records_and_logs_reason(self, tmp_path, log_sink):
        """A failing account stores its reason and logs it."""
        cred_path = tmp_path / "test.json"
        cred_path.write_text(json.dumps({"refreshToken": "rt", "region": "us-east-1"}))
        
        manager = _make_manager(tmp_path, cred_path)
        await manager.load_credentials()
        account_id = str(cred_path.resolve())
        
        exc = _FakeHTTPStatusError("400 Bad Request", _FakeResponse(400, INVALID_GRANT_BODY))
        
        with patch("kiro.account_manager.KiroAuthManager") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth.get_access_token = AsyncMock(side_effect=exc)
            mock_auth_class.return_value = mock_auth
            
            success = await manager._initialize_account(account_id)
        
        assert success is False
        
        reason = manager.get_init_error(account_id)
        assert reason is not None
        assert "invalid_grant" in reason
        assert "HTTP 400" in reason
        
        logged = "".join(str(r) for r in log_sink)
        assert "invalid_grant" in logged
    
    @pytest.mark.asyncio
    async def test_failure_reason_does_not_leak_secrets(self, tmp_path, log_sink):
        """Token values passed through an error never reach log or reason."""
        cred_path = tmp_path / "test.json"
        cred_path.write_text(json.dumps({"refreshToken": "rt", "region": "us-east-1"}))
        
        manager = _make_manager(tmp_path, cred_path)
        await manager.load_credentials()
        account_id = str(cred_path.resolve())
        
        leaky_body = ('{"error":"invalid_grant","refresh_token":"SUPER-SECRET-RT",'
                      '"clientSecret":"SUPER-SECRET-CS"}')
        exc = _FakeHTTPStatusError("400", _FakeResponse(400, leaky_body))
        
        with patch("kiro.account_manager.KiroAuthManager") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth.get_access_token = AsyncMock(side_effect=exc)
            mock_auth_class.return_value = mock_auth
            
            await manager._initialize_account(account_id)
        
        reason = manager.get_init_error(account_id)
        logged = "".join(str(r) for r in log_sink)
        
        assert "SUPER-SECRET-RT" not in reason
        assert "SUPER-SECRET-CS" not in reason
        assert "SUPER-SECRET-RT" not in logged
        assert "SUPER-SECRET-CS" not in logged
        assert "invalid_grant" in reason
    
    @pytest.mark.asyncio
    async def test_unknown_account_records_reason(self, tmp_path):
        """Unknown account IDs get a reason too, contract unchanged."""
        cred_path = tmp_path / "test.json"
        cred_path.write_text(json.dumps({"refreshToken": "rt"}))
        manager = _make_manager(tmp_path, cred_path)
        
        assert await manager._initialize_account("nope") is False
        assert "Unknown account" in manager.get_init_error("nope")
    
    @pytest.mark.asyncio
    async def test_success_records_no_reason(self, tmp_path, mock_list_models_response):
        """A successful account initializes unchanged and clears any reason."""
        cred_path = tmp_path / "test.json"
        cred_path.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1"
        }))
        
        manager = _make_manager(tmp_path, cred_path)
        await manager.load_credentials()
        account_id = str(cred_path.resolve())
        
        with patch("kiro.account_manager.KiroHttpClient") as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            success = await manager._initialize_account(account_id)
        
        assert success is True
        assert manager.get_init_error(account_id) is None
        assert manager._accounts[account_id].auth_manager is not None


# =============================================================================
# Test Class: lifespan aggregates reasons
# =============================================================================

class TestLifespanAggregatesReasons:
    """Tests that startup failure surfaces every account's reason."""
    
    @pytest.mark.asyncio
    async def test_runtime_error_aggregates_reasons(self, tmp_path, monkeypatch):
        """RuntimeError text carries per-account reasons and re-auth guidance."""
        monkeypatch.setattr("main.ACCOUNT_SYSTEM", True)
        monkeypatch.setattr("main.REFRESH_TOKEN", None)
        monkeypatch.setattr("main.KIRO_CREDS_FILE", None)
        monkeypatch.setattr("main.KIRO_CLI_DB_FILE", None)
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": "/enterprise.json", "enabled": True}
        ]))
        monkeypatch.setattr("main.ACCOUNTS_CONFIG_FILE", str(creds_file))
        monkeypatch.setattr("main.ACCOUNTS_STATE_FILE", str(tmp_path / "state.json"))
        
        reason = describe_init_failure(
            _FakeHTTPStatusError("400", _FakeResponse(400, INVALID_GRANT_BODY))
        )
        
        mock_manager = AsyncMock()
        mock_manager._accounts = {"/enterprise.json": MagicMock()}
        mock_manager._current_account_index = 0
        mock_manager._initialize_account = AsyncMock(return_value=False)
        mock_manager.get_init_error = MagicMock(return_value=reason)
        mock_manager._save_state = AsyncMock()
        mock_manager.save_state_periodically = AsyncMock()
        
        with patch("main.AccountManager", return_value=mock_manager):
            with patch("main.httpx.AsyncClient") as mock_client_class:
                mock_client_class.return_value = AsyncMock()
                
                from main import lifespan, app
                
                with pytest.raises(RuntimeError) as exc_info:
                    async with lifespan(app):
                        pass
        
        message = str(exc_info.value)
        assert "/enterprise.json" in message
        assert "invalid_grant" in message
        assert "re-authenticated" in message
