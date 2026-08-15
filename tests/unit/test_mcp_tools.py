# -*- coding: utf-8 -*-

"""
Tests for MCP Tools Support (WebSearch).

Tests cover:
- ID generation
- MCP API calls
- Search summary generation
- Query extraction from messages
- Native web_search handler (Path A)
- SSE emulation (Anthropic and OpenAI formats)
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from kiro.mcp_tools import (
    generate_random_id,
    call_kiro_mcp_api,
    generate_search_summary,
    extract_query_from_messages,
    handle_native_web_search,
    generate_anthropic_web_search_sse,
    generate_openai_web_search_sse
)


# ==================================================================================================
# Tests for ID Generation
# ==================================================================================================

class TestIDGeneration:
    """Tests for random ID generation."""
    
    def test_generate_random_id_length(self):
        """
        What it does: Verifies ID generation with exact length.
        Purpose: Ensure generate_random_id returns correct length.
        """
        print("Setup: Generating IDs of different lengths...")
        
        print("Action: Generate ID of length 22...")
        id_22 = generate_random_id(22)
        print(f"Comparing length: Expected 22, Got {len(id_22)}")
        assert len(id_22) == 22
        
        print("Action: Generate ID of length 8...")
        id_8 = generate_random_id(8)
        print(f"Comparing length: Expected 8, Got {len(id_8)}")
        assert len(id_8) == 8
        
        print("Action: Generate ID of length 100...")
        id_100 = generate_random_id(100)
        print(f"Comparing length: Expected 100, Got {len(id_100)}")
        assert len(id_100) == 100
    
    def test_generate_random_id_alphanumeric(self):
        """
        What it does: Verifies ID contains only alphanumeric characters.
        Purpose: Ensure no special characters in generated IDs.
        """
        print("Setup: Generating large ID to test character set...")
        
        print("Action: Generate ID of length 1000...")
        random_id = generate_random_id(1000)
        
        print(f"Checking if alphanumeric: {random_id[:50]}...")
        assert random_id.isalnum()
    
    def test_generate_random_id_uniqueness(self):
        """
        What it does: Verifies IDs are unique (probabilistically).
        Purpose: Ensure randomness works correctly.
        """
        print("Setup: Generating multiple IDs...")
        
        print("Action: Generate 100 IDs of length 22...")
        ids = [generate_random_id(22) for _ in range(100)]
        
        print(f"Comparing uniqueness: Generated {len(ids)} IDs, unique: {len(set(ids))}")
        assert len(set(ids)) == len(ids)  # All should be unique


# ==================================================================================================
# Tests for MCP API Call
# ==================================================================================================

class TestCallKiroMCPAPI:
    """Tests for MCP API calls."""
    
    @pytest.mark.asyncio
    async def test_mcp_api_success(self, mock_auth_manager):
        """
        What it does: Verifies successful MCP API call and result parsing.
        Purpose: Ensure MCP API integration works correctly.
        """
        print("Setup: Mocking successful MCP API response...")
        query = "Python tutorials"
        
        # Mock MCP response (CRITICAL: result.content[0].text is JSON STRING)
        mock_response_data = {
            "id": "web_search_tooluse_abc123_1234567890_xyz",
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [
                            {
                                "title": "Python Tutorial",
                                "url": "https://python.org",
                                "snippet": "Learn Python programming",
                                "publishedDate": 1700000000000
                            }
                        ],
                        "totalResults": 1,
                        "query": "Python tutorials"
                    })
                }],
                "isError": False
            }
        }
        
        # Mock httpx.AsyncClient - CRITICAL: json() must be async
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_response_data)
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing tool_use_id: Got '{tool_use_id}'")
        assert tool_use_id is not None
        assert tool_use_id.startswith("srvtoolu_")
        
        print(f"Comparing results: Got {results}")
        assert results is not None
        assert results["totalResults"] == 1
        assert results["results"][0]["title"] == "Python Tutorial"
        assert results["results"][0]["url"] == "https://python.org"
    
    @pytest.mark.asyncio
    async def test_mcp_api_error_response(self, mock_auth_manager):
        """
        What it does: Verifies handling of MCP API error response.
        Purpose: Ensure errors are handled gracefully.
        """
        print("Setup: Mocking MCP API error response...")
        query = "test"
        
        # Mock error response
        mock_response_data = {
            "id": "web_search_tooluse_abc123_1234567890_xyz",
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid request"}
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_response_data)
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None
    
    @pytest.mark.asyncio
    async def test_mcp_api_http_error(self, mock_auth_manager):
        """
        What it does: Verifies handling of HTTP errors from MCP API.
        Purpose: Ensure non-200 status codes are handled.
        """
        print("Setup: Mocking HTTP 500 error...")
        query = "test"
        
        mock_response = Mock()
        mock_response.status_code = 500
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None
    
    @pytest.mark.asyncio
    async def test_mcp_api_timeout(self, mock_auth_manager):
        """
        What it does: Verifies handling of MCP API timeout.
        Purpose: Ensure timeouts are handled gracefully.
        """
        print("Setup: Mocking timeout exception...")
        query = "test"
        
        import httpx
        
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None
    
    @pytest.mark.asyncio
    async def test_mcp_api_json_decode_error(self, mock_auth_manager):
        """
        What it does: Verifies handling of malformed JSON in MCP response.
        Purpose: Ensure JSON parsing errors are handled.
        """
        print("Setup: Mocking malformed JSON response...")
        query = "test"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        
        mock_post = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = mock_post
        
        print("Action: Calling call_kiro_mcp_api...")
        with patch("kiro.mcp_tools.httpx.AsyncClient", return_value=mock_client):
            tool_use_id, results = await call_kiro_mcp_api(query, mock_auth_manager)
        
        print(f"Comparing result: Expected (None, None), Got ({tool_use_id}, {results})")
        assert tool_use_id is None
        assert results is None


# ==================================================================================================
# Tests for Search Summary Generation
# ==================================================================================================

class TestGenerateSearchSummary:
    """Tests for search summary formatting."""
    
    def test_generate_summary_with_results(self):
        """
        What it does: Verifies summary formatting with results.
        Purpose: Ensure XML tags and proper formatting.
        """
        print("Setup: Creating mock search results...")
        query = "Python"
        results = {
            "results": [
                {
                    "title": "Python.org",
                    "url": "https://python.org",
                    "snippet": "Official Python website with tutorials",
                    "publishedDate": 1700000000000
                },
                {
                    "title": "Python Tutorial",
                    "url": "https://docs.python.org",
                    "snippet": "Complete Python documentation",
                    "publishedDate": None  # No date
                }
            ],
            "totalResults": 2
        }
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking XML tags...")
        assert "<web_search>" in summary
        assert "</web_search>" in summary
        
        print(f"Checking query in summary...")
        assert "Python" in summary
        
        print(f"Checking first result...")
        assert "Python.org" in summary
        assert "https://python.org" in summary
        assert "Official Python website with tutorials" in summary
        
        print(f"Checking second result...")
        assert "Python Tutorial" in summary
        assert "https://docs.python.org" in summary
        assert "Complete Python documentation" in summary
    
    def test_generate_summary_no_results(self):
        """
        What it does: Verifies summary with empty results list.
        Purpose: Ensure empty results are handled gracefully.
        """
        print("Setup: Creating empty results...")
        query = "nonexistent"
        results = {"results": [], "totalResults": 0}
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking XML tags...")
        assert "<web_search>" in summary
        assert "</web_search>" in summary
        
        print(f"Checking query in summary...")
        assert "nonexistent" in summary
        
        print(f"Summary content: {repr(summary)}")
        # Empty results list produces empty content between tags (no "No results found")
        assert "Search results for" in summary
    
    def test_generate_summary_malformed_results(self):
        """
        What it does: Verifies handling of malformed results.
        Purpose: Ensure graceful handling of invalid data.
        """
        print("Setup: Creating malformed results...")
        query = "test"
        results = {"invalid": "structure"}
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking for 'No results found'...")
        assert "No results found" in summary
    
    def test_generate_summary_date_formatting(self):
        """
        What it does: Verifies date formatting from milliseconds timestamp.
        Purpose: Ensure publishedDate is converted correctly.
        """
        print("Setup: Creating result with timestamp...")
        query = "test"
        # 1700000000000 ms = 2023-11-14 22:13:20 UTC
        results = {
            "results": [{
                "title": "Test",
                "url": "https://test.com",
                "snippet": "Test snippet",
                "publishedDate": 1700000000000
            }],
            "totalResults": 1
        }
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking date format...")
        # Should contain formatted date like "14 Nov 2023"
        assert "Nov 2023" in summary or "Ноя 2023" in summary  # Depends on locale
    
    def test_generate_summary_full_snippet_no_truncation(self):
        """
        What it does: Verifies snippets are NOT truncated.
        Purpose: Ensure model gets full information.
        """
        print("Setup: Creating result with long snippet...")
        query = "test"
        long_snippet = "A" * 1000  # 1000 characters
        results = {
            "results": [{
                "title": "Test",
                "url": "https://test.com",
                "snippet": long_snippet,
                "publishedDate": None
            }],
            "totalResults": 1
        }
        
        print("Action: Generating summary...")
        summary = generate_search_summary(query, results)
        
        print(f"Checking snippet is NOT truncated...")
        assert long_snippet in summary
        assert len(long_snippet) == 1000  # Full length preserved


# ==================================================================================================
# Tests for Query Extraction
# ==================================================================================================

class TestExtractQueryFromMessages:
    """Tests for query extraction from messages."""
    
    def test_extract_query_anthropic_string_content(self):
        """
        What it does: Extracts query from Anthropic string content.
        Purpose: Ensure simple string messages work.
        """
        print("Setup: Creating Anthropic message with string content...")
        from kiro.models_anthropic import AnthropicMessage
        messages = [AnthropicMessage(role="user", content="Search for Python tutorials")]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Search for Python tutorials', Got '{query}'")
        assert query == "Search for Python tutorials"
    
    def test_extract_query_anthropic_list_content(self):
        """
        What it does: Extracts query from Anthropic list content.
        Purpose: Ensure content blocks work.
        """
        print("Setup: Creating Anthropic message with list content...")
        from kiro.models_anthropic import AnthropicMessage, TextContentBlock
        messages = [AnthropicMessage(
            role="user",
            content=[TextContentBlock(type="text", text="Python tutorials")]
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Python tutorials', Got '{query}'")
        assert query == "Python tutorials"
    
    def test_extract_query_with_prefix(self):
        """
        What it does: Removes 'Perform a web search for the query:' prefix.
        Purpose: Ensure prefix is stripped correctly.
        """
        print("Setup: Creating message with prefix...")
        from kiro.models_anthropic import AnthropicMessage
        messages = [AnthropicMessage(
            role="user",
            content="Perform a web search for the query: Python"
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Python', Got '{query}'")
        assert query == "Python"
    
    def test_extract_query_empty_messages(self):
        """
        What it does: Handles empty messages list.
        Purpose: Ensure None is returned for empty input.
        """
        print("Setup: Creating empty messages list...")
        messages = []
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected None, Got {query}")
        assert query is None
    
    def test_extract_query_no_text_content(self):
        """
        What it does: Handles messages without text content.
        Purpose: Ensure None is returned for non-text messages.
        """
        print("Setup: Creating message with image content...")
        from kiro.models_anthropic import AnthropicMessage, ImageContentBlock, Base64ImageSource
        messages = [AnthropicMessage(
            role="user",
            content=[ImageContentBlock(
                type="image",
                source=Base64ImageSource(
                    type="base64",
                    media_type="image/png",
                    data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                )
            )]
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected None or empty, Got '{query}'")
        assert query is None or query == ""
    
    def test_extract_query_multiple_text_blocks(self):
        """
        What it does: Concatenates multiple text blocks.
        Purpose: Ensure all text is extracted.
        """
        print("Setup: Creating message with multiple text blocks...")
        from kiro.models_anthropic import AnthropicMessage, TextContentBlock
        messages = [AnthropicMessage(
            role="user",
            content=[
                TextContentBlock(type="text", text="Search for "),
                TextContentBlock(type="text", text="Python tutorials")
            ]
        )]
        
        print("Action: Extracting query...")
        query = extract_query_from_messages(messages, "anthropic")
        
        print(f"Comparing query: Expected 'Search for Python tutorials', Got '{query}'")
        assert query == "Search for Python tutorials"


# ==================================================================================================
# Tests for SSE Emulation
# ==================================================================================================

class TestAnthropicSSEEmulation:
    """Tests for Anthropic SSE stream generation."""
    
    @pytest.mark.asyncio
    async def test_generate_anthropic_sse_structure(self):
        """
        What it does: Verifies Anthropic SSE event structure.
        Purpose: Ensure all 11 events are generated correctly.
        """
        print("Setup: Preparing test data...")
        model = "claude-sonnet-4"
        query = "Python"
        tool_use_id = "srvtoolu_test123"
        results = {
            "results": [{"title": "Test", "url": "https://test.com", "snippet": "Test"}],
            "totalResults": 1
        }
        input_tokens = 100
        
        print("Action: Generating SSE stream...")
        events = []
        async for event in generate_anthropic_web_search_sse(model, query, tool_use_id, results, input_tokens):
            events.append(event)
        
        print(f"Comparing event count: Got {len(events)} events")
        assert len(events) >= 11  # At least 11 events (may have more text_delta chunks)
        
        print("Checking event types...")
        event_types = []
        for event in events:
            if "event:" in event:
                event_type = event.split("event:")[1].split("\n")[0].strip()
                event_types.append(event_type)
        
        print(f"Event types: {event_types}")
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types


class TestOpenAISSEEmulation:
    """Tests for OpenAI SSE stream generation."""
    
    @pytest.mark.asyncio
    async def test_generate_openai_sse_structure(self):
        """
        What it does: Verifies OpenAI SSE event structure.
        Purpose: Ensure OpenAI format is correct.
        """
        print("Setup: Preparing test data...")
        model = "claude-sonnet-4"
        query = "Python"
        tool_use_id = "srvtoolu_test123"
        results = {
            "results": [{"title": "Test", "url": "https://test.com", "snippet": "Test"}],
            "totalResults": 1
        }
        input_tokens = 100
        
        print("Action: Generating SSE stream...")
        chunks = []
        async for chunk in generate_openai_web_search_sse(model, query, tool_use_id, results, input_tokens):
            chunks.append(chunk)
        
        print(f"Comparing chunk count: Got {len(chunks)} chunks")
        assert len(chunks) >= 3  # At least: role, content chunks, finish + [DONE]
        
        print("Checking for [DONE] marker...")
        assert any("[DONE]" in chunk for chunk in chunks)
        
        print("Checking for role delta (flexible matching)...")
        assert any('"role"' in chunk and '"assistant"' in chunk for chunk in chunks)
        
        print("Checking for finish_reason (flexible matching)...")
        assert any('"finish_reason"' in chunk and '"stop"' in chunk for chunk in chunks)
        
        print("Checking for data: prefix...")
        assert any(chunk.startswith("data:") for chunk in chunks)
        
        print("Checking for usage information...")
        assert any('"usage"' in chunk for chunk in chunks)



# ==================================================================================================
# Tests for WebSearch Continuation / Synthesis (issue #258)
# ==================================================================================================

from kiro.mcp_tools import (
    MAX_WEB_SEARCH_CONTINUATION_ROUNDS,
    has_search_results,
    build_synthesis_prompt,
    stream_web_search_synthesis,
    collect_web_search_synthesis,
)


SAMPLE_RESULTS = {
    "results": [{
        "title": "Python 3.13 release notes",
        "url": "https://python.org/3.13",
        "snippet": "Python 3.13 introduces a new REPL.",
    }],
    "totalResults": 1,
}


class _FakeEvent:
    """Minimal stand-in for KiroEvent."""

    def __init__(self, type_, content=None):
        self.type = type_
        self.content = content


def _make_auth_manager():
    auth_manager = Mock()
    auth_manager.api_host = "https://kiro.example"
    auth_manager.profile_arn = "arn:aws:test"
    auth_manager.get_access_token = AsyncMock(return_value="token")
    return auth_manager


def _patch_upstream(events, status_code=200, request_error=None, captured=None):
    """
    Patch KiroHttpClient + parse_kiro_stream so the continuation call is fully offline.

    Returns a context manager tuple to be used with `with`.
    """
    response = Mock()
    response.status_code = status_code

    async def fake_request(method, url, json_data=None, params=None, stream=False):
        if captured is not None:
            captured["method"] = method
            captured["url"] = url
            captured["payload"] = json_data
            captured["stream"] = stream
        if request_error is not None:
            raise request_error
        return response

    fake_client = Mock()
    fake_client.request_with_retry = fake_request
    fake_client.close = AsyncMock()

    async def fake_parse(resp, *args, **kwargs):
        for ev in events:
            yield ev

    client_patch = patch("kiro.http_client.KiroHttpClient", return_value=fake_client)
    parse_patch = patch("kiro.streaming_core.parse_kiro_stream", fake_parse)
    return client_patch, parse_patch, fake_client


class TestSynthesisHelpers:
    """Tests for the small pure helpers backing the continuation."""

    def test_has_search_results_true(self):
        print("Action: Checking non-empty results...")
        assert has_search_results(SAMPLE_RESULTS) is True

    def test_has_search_results_false_cases(self):
        print("Action: Checking None / empty / missing key...")
        assert has_search_results(None) is False
        assert has_search_results({}) is False
        assert has_search_results({"results": []}) is False
        assert has_search_results({"totalResults": 0}) is False

    def test_build_synthesis_prompt_carries_results(self):
        """
        What it does: Verifies the continuation prompt contains the search results.
        Purpose: This is the actual feedback channel fixing issue #258.
        """
        print("Action: Building synthesis prompt...")
        prompt = build_synthesis_prompt("python 3.13", SAMPLE_RESULTS)

        print("Checking query, URL and snippet are present...")
        assert "python 3.13" in prompt
        assert "https://python.org/3.13" in prompt
        assert "Python 3.13 introduces a new REPL." in prompt

        print("Checking the model is instructed to answer, not search again...")
        assert "Do not perform another search" in prompt

    def test_continuation_cap_default_is_bounded(self):
        print("Action: Checking the cap constant...")
        assert isinstance(MAX_WEB_SEARCH_CONTINUATION_ROUNDS, int)
        assert 1 <= MAX_WEB_SEARCH_CONTINUATION_ROUNDS <= 3


class TestStreamWebSearchSynthesis:
    """Tests for the continuation call itself. No live API calls are made."""

    @pytest.mark.asyncio
    async def test_results_are_fed_back_and_synthesis_occurs(self):
        """
        What it does: Runs the continuation and collects the model's answer.
        Purpose: Core issue #258 fix - the turn no longer ends at the summary.
        """
        print("Setup: Patching upstream with a two-chunk content stream...")
        captured = {}
        events = [
            _FakeEvent("content", "Python 3.13 "),
            _FakeEvent("usage", None),
            _FakeEvent("content", "adds a new REPL."),
        ]
        client_patch, parse_patch, fake_client = _patch_upstream(events, captured=captured)

        print("Action: Streaming synthesis...")
        with client_patch, parse_patch:
            chunks = [
                c async for c in stream_web_search_synthesis(
                    "claude-sonnet-4", "python 3.13", SAMPLE_RESULTS, _make_auth_manager()
                )
            ]

        print(f"Comparing output: Got {chunks}")
        assert "".join(chunks) == "Python 3.13 adds a new REPL."

        print("Checking the upstream call carried the search results...")
        payload_json = json.dumps(captured["payload"])
        assert "https://python.org/3.13" in payload_json
        assert captured["url"].endswith("/generateAssistantResponse")
        assert captured["stream"] is True

        print("Checking the HTTP client was closed...")
        fake_client.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_continuation_payload_declares_no_tools(self):
        """
        What it does: Verifies the continuation request carries no tools.
        Purpose: Structural loop guard - the model cannot request another web_search.
        """
        print("Setup: Patching upstream...")
        captured = {}
        client_patch, parse_patch, _ = _patch_upstream(
            [_FakeEvent("content", "ok")], captured=captured
        )

        print("Action: Streaming synthesis...")
        with client_patch, parse_patch:
            async for _ in stream_web_search_synthesis(
                "claude-sonnet-4", "python", SAMPLE_RESULTS, _make_auth_manager()
            ):
                pass

        print("Checking no tool specification is present in the payload...")
        payload_json = json.dumps(captured["payload"])
        assert "toolSpecification" not in payload_json

    @pytest.mark.asyncio
    async def test_continuation_cap_is_enforced(self):
        """
        What it does: Calls with an exhausted round budget.
        Purpose: Ensure no unbounded continuation loop - no upstream call at all.
        """
        print("Setup: Patching upstream (must not be used)...")
        captured = {}
        client_patch, parse_patch, _ = _patch_upstream(
            [_FakeEvent("content", "should not appear")], captured=captured
        )

        print("Action: Streaming with round_index == max_rounds...")
        with client_patch, parse_patch:
            chunks = [
                c async for c in stream_web_search_synthesis(
                    "claude-sonnet-4", "python", SAMPLE_RESULTS, _make_auth_manager(),
                    round_index=1, max_rounds=1
                )
            ]

        print(f"Comparing output: Expected [], Got {chunks}")
        assert chunks == []

        print("Checking no upstream request was made...")
        assert captured == {}

    @pytest.mark.asyncio
    async def test_continuation_cap_enforced_at_default(self):
        """
        What it does: Verifies the default cap blocks a second round.
        Purpose: The default configuration is bounded without any caller effort.
        """
        print("Action: Streaming with round_index == default cap...")
        chunks = [
            c async for c in stream_web_search_synthesis(
                "claude-sonnet-4", "python", SAMPLE_RESULTS, _make_auth_manager(),
                round_index=MAX_WEB_SEARCH_CONTINUATION_ROUNDS
            )
        ]
        print(f"Comparing output: Expected [], Got {chunks}")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_empty_results_degrade_gracefully(self):
        """
        What it does: Runs synthesis with empty / failed search results.
        Purpose: Must return immediately, no hang, no exception, no upstream call.
        """
        print("Action: Streaming with empty and None results...")
        for bad in (None, {}, {"results": []}):
            chunks = [
                c async for c in stream_web_search_synthesis(
                    "claude-sonnet-4", "python", bad, _make_auth_manager()
                )
            ]
            print(f"Comparing output for {bad!r}: Expected [], Got {chunks}")
            assert chunks == []

    @pytest.mark.asyncio
    async def test_missing_auth_manager_degrades_gracefully(self):
        print("Action: Streaming without an auth manager...")
        chunks = [
            c async for c in stream_web_search_synthesis(
                "claude-sonnet-4", "python", SAMPLE_RESULTS, None
            )
        ]
        print(f"Comparing output: Expected [], Got {chunks}")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_upstream_non_200_degrades_gracefully(self):
        """
        What it does: Upstream returns 500 for the continuation.
        Purpose: Must yield nothing (client still gets the summary), never raise.
        """
        print("Setup: Patching upstream with status 500...")
        client_patch, parse_patch, fake_client = _patch_upstream(
            [_FakeEvent("content", "unused")], status_code=500
        )

        print("Action: Streaming synthesis...")
        with client_patch, parse_patch:
            chunks = [
                c async for c in stream_web_search_synthesis(
                    "claude-sonnet-4", "python", SAMPLE_RESULTS, _make_auth_manager()
                )
            ]

        print(f"Comparing output: Expected [], Got {chunks}")
        assert chunks == []
        fake_client.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_upstream_exception_degrades_gracefully(self):
        """
        What it does: Upstream raises during the continuation request.
        Purpose: Failure must be swallowed so the client never sees a 500.
        """
        print("Setup: Patching upstream to raise...")
        client_patch, parse_patch, fake_client = _patch_upstream(
            [], request_error=RuntimeError("boom")
        )

        print("Action: Streaming synthesis...")
        with client_patch, parse_patch:
            chunks = [
                c async for c in stream_web_search_synthesis(
                    "claude-sonnet-4", "python", SAMPLE_RESULTS, _make_auth_manager()
                )
            ]

        print(f"Comparing output: Expected [], Got {chunks}")
        assert chunks == []
        fake_client.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_collect_web_search_synthesis_joins_chunks(self):
        print("Setup: Patching upstream with two content events...")
        client_patch, parse_patch, _ = _patch_upstream([
            _FakeEvent("content", "A"),
            _FakeEvent("content", "B"),
        ])

        print("Action: Collecting synthesis...")
        with client_patch, parse_patch:
            text = await collect_web_search_synthesis(
                "claude-sonnet-4", "python", SAMPLE_RESULTS, _make_auth_manager()
            )

        print(f"Comparing text: Expected 'AB', Got '{text}'")
        assert text == "AB"

    @pytest.mark.asyncio
    async def test_collect_returns_empty_string_on_failure(self):
        print("Action: Collecting with no results...")
        text = await collect_web_search_synthesis(
            "claude-sonnet-4", "python", None, _make_auth_manager()
        )
        print(f"Comparing text: Expected '', Got '{text}'")
        assert text == ""


class TestSSEWithSynthesis:
    """Tests that the synthesised answer reaches the client stream."""

    @staticmethod
    async def _text_stream(*parts):
        for p in parts:
            yield p

    @pytest.mark.asyncio
    async def test_anthropic_sse_streams_synthesis_block(self):
        """
        What it does: Passes a synthesis stream to the Anthropic SSE emitter.
        Purpose: The client must receive the model's answer after the summary.
        """
        print("Setup: Preparing synthesis stream...")
        stream = self._text_stream("The answer ", "is 42.")

        print("Action: Generating SSE...")
        events = []
        async for ev in generate_anthropic_web_search_sse(
            "claude-sonnet-4", "python", "srvtoolu_x", SAMPLE_RESULTS, 100,
            synthesis_stream=stream
        ):
            events.append(ev)

        joined = "".join(events)
        print("Checking synthesis text is present...")
        assert "The answer " in joined
        assert "is 42." in joined

        print("Checking a dedicated text block (index 3) was opened and closed...")
        assert '"index": 3' in joined
        assert joined.count('"index": 3') >= 3  # start + 2 deltas + stop

        print("Checking message_stop still terminates the stream...")
        assert "message_stop" in events[-1]

    @pytest.mark.asyncio
    async def test_anthropic_sse_no_empty_block_when_synthesis_empty(self):
        """
        What it does: Synthesis yields nothing (failed search continuation).
        Purpose: Stream must be byte-identical to the pre-fix behaviour.
        """
        print("Setup: Empty synthesis stream...")
        empty = self._text_stream()

        print("Action: Generating SSE with and without synthesis...")
        with_synth = [
            e async for e in generate_anthropic_web_search_sse(
                "claude-sonnet-4", "python", "srvtoolu_x", SAMPLE_RESULTS, 100,
                synthesis_stream=empty
            )
        ]
        without = [
            e async for e in generate_anthropic_web_search_sse(
                "claude-sonnet-4", "python", "srvtoolu_x", SAMPLE_RESULTS, 100
            )
        ]

        print("Checking no index-3 block was emitted...")
        assert '"index": 3' not in "".join(with_synth)

        print("Comparing event counts (message_id differs, count must not)...")
        assert len(with_synth) == len(without)

    @pytest.mark.asyncio
    async def test_openai_sse_streams_synthesis_content(self):
        """
        What it does: Passes a synthesis stream to the OpenAI SSE emitter.
        Purpose: OpenAI clients also get the synthesised answer.
        """
        print("Setup: Preparing synthesis stream...")
        stream = self._text_stream("Synthesised answer.")

        print("Action: Generating SSE...")
        chunks = []
        async for c in generate_openai_web_search_sse(
            "claude-sonnet-4", "python", "srvtoolu_x", SAMPLE_RESULTS, 100,
            synthesis_stream=stream
        ):
            chunks.append(c)

        joined = "".join(chunks)
        print("Checking synthesis content and terminator...")
        assert "Synthesised answer." in joined
        assert chunks[-1].strip() == "data: [DONE]"
        assert '"finish_reason": "stop"' in joined or '"finish_reason":"stop"' in joined

    @pytest.mark.asyncio
    async def test_sse_survives_failing_synthesis_stream(self):
        """
        What it does: Synthesis stream raises mid-iteration.
        Purpose: The client stream must still terminate cleanly (no hang, no 500).
        """
        async def broken():
            yield "partial "
            raise RuntimeError("upstream died")

        print("Action: Generating Anthropic SSE with a broken synthesis stream...")
        events = [
            e async for e in generate_anthropic_web_search_sse(
                "claude-sonnet-4", "python", "srvtoolu_x", SAMPLE_RESULTS, 100,
                synthesis_stream=broken()
            )
        ]
        joined = "".join(events)
        print("Checking partial text delivered and stream terminated...")
        assert "partial " in joined
        assert "message_stop" in events[-1]

        print("Action: Same for OpenAI...")
        chunks = [
            c async for c in generate_openai_web_search_sse(
                "claude-sonnet-4", "python", "srvtoolu_x", SAMPLE_RESULTS, 100,
                synthesis_stream=broken()
            )
        ]
        assert chunks[-1].strip() == "data: [DONE]"


class TestHandleNativeWebSearchSynthesis:
    """End-to-end (offline) checks of Path A with synthesis wired in."""

    @staticmethod
    def _request_data(stream):
        from kiro.models_anthropic import AnthropicMessagesRequest
        return AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1024,
            stream=stream,
            messages=[{"role": "user", "content": "python 3.13 news"}],
        )

    @pytest.mark.asyncio
    async def test_non_streaming_appends_synthesis_block(self):
        """
        What it does: Path A non-streaming with a working continuation.
        Purpose: Response carries the synthesised answer, not just the dump.
        """
        print("Setup: Patching MCP API and upstream continuation...")
        client_patch, parse_patch, _ = _patch_upstream([_FakeEvent("content", "Synth answer.")])

        with patch(
            "kiro.mcp_tools.call_kiro_mcp_api",
            AsyncMock(return_value=("srvtoolu_x", SAMPLE_RESULTS)),
        ), client_patch, parse_patch:
            print("Action: Calling handle_native_web_search...")
            response = await handle_native_web_search(
                Mock(), self._request_data(stream=False), _make_auth_manager(),
                api_format="anthropic"
            )

        body = json.loads(response.body)
        texts = [b["text"] for b in body["content"] if b["type"] == "text"]
        print(f"Checking synthesis text present in: {texts}")
        assert any("Synth answer." in t for t in texts)

        print("Checking the raw summary is still present...")
        assert any("<web_search>" in t for t in texts)

    @pytest.mark.asyncio
    async def test_non_streaming_degrades_when_continuation_fails(self):
        """
        What it does: Continuation upstream fails; search itself succeeded.
        Purpose: Must return 200 with the summary, not a 500.
        """
        print("Setup: Patching MCP API OK, continuation raising...")
        client_patch, parse_patch, _ = _patch_upstream([], request_error=RuntimeError("boom"))

        with patch(
            "kiro.mcp_tools.call_kiro_mcp_api",
            AsyncMock(return_value=("srvtoolu_x", SAMPLE_RESULTS)),
        ), client_patch, parse_patch:
            print("Action: Calling handle_native_web_search...")
            response = await handle_native_web_search(
                Mock(), self._request_data(stream=False), _make_auth_manager(),
                api_format="anthropic"
            )

        print(f"Comparing status: Expected 200, Got {response.status_code}")
        assert response.status_code == 200
        body = json.loads(response.body)
        texts = [b["text"] for b in body["content"] if b["type"] == "text"]
        assert any("<web_search>" in t for t in texts)

    @pytest.mark.asyncio
    async def test_failed_search_still_returns_error_not_hang(self):
        """
        What it does: MCP search itself fails (results None).
        Purpose: Existing contract preserved - a clean 500 error body, no hang.
        """
        print("Setup: Patching MCP API to fail...")
        with patch(
            "kiro.mcp_tools.call_kiro_mcp_api",
            AsyncMock(return_value=(None, None)),
        ):
            print("Action: Calling handle_native_web_search...")
            response = await handle_native_web_search(
                Mock(), self._request_data(stream=True), _make_auth_manager(),
                api_format="anthropic"
            )

        print(f"Comparing status: Expected 500, Got {response.status_code}")
        assert response.status_code == 500
