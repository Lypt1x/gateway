# -*- coding: utf-8 -*-
"""
Tests for the MCP web_search request shape (issues #254, #173, #231, #168).

All offline: httpx.AsyncClient is mocked, no network calls are made.

Empirically required by the real /mcp endpoint:
- profileArn at the JSON-RPC BODY ROOT (in params => still 400)
- Kiro client-identity headers, User-Agent in particular (ARN alone => 403)
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch

from kiro.mcp_tools import call_kiro_mcp_api


def _mock_client(status_code=200, body_text="", payload=None):
    """Build a mocked httpx.AsyncClient plus its captured post mock."""
    if payload is None:
        payload = {
            "id": "web_search_tooluse_x",
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [{"title": "T", "url": "https://e.org", "snippet": "s"}],
                        "totalResults": 1,
                        "query": "q",
                    }),
                }],
                "isError": False,
            },
        }

    response = Mock()
    response.status_code = status_code
    response.json = Mock(return_value=payload)
    response.text = body_text

    post = AsyncMock(return_value=response)
    client = AsyncMock()
    client.__aenter__.return_value.post = post
    return client, post


class TestMCPProfileArn:
    """profileArn placement and omission."""

    @pytest.mark.asyncio
    async def test_profile_arn_at_body_root_not_in_params(self, mock_auth_manager):
        client, post = _mock_client()
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=client):
            tool_use_id, results = await call_kiro_mcp_api("q", mock_auth_manager)

        assert tool_use_id is not None and results is not None
        body = post.call_args.kwargs["json"]
        assert body["profileArn"] == mock_auth_manager.profile_arn
        assert "profileArn" not in body["params"]
        assert "profileArn" not in body["params"]["arguments"]
        # Standard JSON-RPC siblings intact
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_arn_absent_key_omitted_and_warning_logged(self, mock_auth_manager, caplog):
        mock_auth_manager._profile_arn = None
        client, post = _mock_client()

        warnings = []
        with patch("kiro.mcp_tools.PROFILE_ARN", ""), \
             patch("kiro.mcp_tools.logger.warning", side_effect=lambda m, *a, **k: warnings.append(str(m))), \
             patch("kiro.mcp_tools.httpx.AsyncClient", return_value=client):
            await call_kiro_mcp_api("q", mock_auth_manager)

        body = post.call_args.kwargs["json"]
        assert "profileArn" not in body
        assert any("profileArn" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_falls_back_to_config_profile_arn(self, mock_auth_manager):
        mock_auth_manager._profile_arn = None
        client, post = _mock_client()
        with patch("kiro.mcp_tools.PROFILE_ARN", "arn:from:config"), \
             patch("kiro.mcp_tools.httpx.AsyncClient", return_value=client):
            await call_kiro_mcp_api("q", mock_auth_manager)

        assert post.call_args.kwargs["json"]["profileArn"] == "arn:from:config"


class TestMCPHeaders:
    """Kiro identity headers, adapted for the JSON-RPC /mcp endpoint."""

    @pytest.mark.asyncio
    async def test_headers_shape(self, mock_auth_manager):
        client, post = _mock_client()
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=client):
            await call_kiro_mcp_api("q", mock_auth_manager)

        headers = post.call_args.kwargs["headers"]
        assert "Kiro" in headers["User-Agent"]
        assert headers["Content-Type"] == "application/json"
        assert "x-amz-target" not in headers
        assert headers["x-amzn-codewhisperer-optout"] == "false"
        assert headers["x-amzn-kiro-agent-mode"] == "vibe"
        assert "x-amz-user-agent" in headers
        assert headers["Authorization"].startswith("Bearer ")


class TestMCPErrorSurfacing:
    """Non-200 responses must surface the server's explanation."""

    @pytest.mark.asyncio
    async def test_non_200_logs_response_body(self, mock_auth_manager):
        body = '{"message":"profileArn is required for this request."}'
        client, _ = _mock_client(status_code=400, body_text=body)

        errors = []
        with patch("kiro.mcp_tools.logger.error", side_effect=lambda m, *a, **k: errors.append(str(m))), \
             patch("kiro.mcp_tools.httpx.AsyncClient", return_value=client):
            result = await call_kiro_mcp_api("q", mock_auth_manager)

        assert result == (None, None)
        joined = "\n".join(errors)
        assert "400" in joined
        assert "profileArn is required for this request." in joined

    @pytest.mark.asyncio
    async def test_no_token_or_authorization_in_logs(self, mock_auth_manager):
        client, _ = _mock_client(status_code=403, body_text='{"message":"User is not authorized"}')

        messages = []
        sink = lambda m, *a, **k: messages.append(str(m))
        with patch("kiro.mcp_tools.logger.error", side_effect=sink), \
             patch("kiro.mcp_tools.logger.warning", side_effect=sink), \
             patch("kiro.mcp_tools.logger.debug", side_effect=sink), \
             patch("kiro.mcp_tools.httpx.AsyncClient", return_value=client):
            await call_kiro_mcp_api("q", mock_auth_manager)

        joined = "\n".join(messages)
        token = await mock_auth_manager.get_access_token()
        assert token not in joined
        assert "Bearer " not in joined
        assert "Authorization" not in joined
