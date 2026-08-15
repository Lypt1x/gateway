# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Unified Account System for Kiro Gateway.

Manages multiple Kiro accounts with intelligent failover, sticky behavior,
and circuit breaker pattern for reliability.

Key features:
- Lazy initialization (only first working account at startup)
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- Probabilistic retry for "dead" accounts
- TTL-based model cache refresh (only when using account)
- Atomic state persistence
"""

import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from loguru import logger

from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver, normalize_model_name
from kiro.config import (
    HIDDEN_MODELS,
    MODEL_ALIASES,
    HIDDEN_FROM_LIST,
    ACCOUNT_RECOVERY_TIMEOUT,
    ACCOUNT_MAX_BACKOFF_MULTIPLIER,
    ACCOUNT_PROBABILISTIC_RETRY_CHANCE,
    ACCOUNT_CACHE_TTL,
    STATE_SAVE_INTERVAL_SECONDS,
    FALLBACK_MODELS,
    MODEL_CACHE_TTL,
    MODEL_DISCOVERY,
    MODEL_DISCOVERY_TIMEOUT,
    REGION,
    get_kiro_control_plane_host,
)
from kiro.utils import get_kiro_headers
from kiro.account_errors import ErrorType
from kiro.http_client import KiroHttpClient


# --- Model-discovery pagination caps -------------------------------------------------
# ListAvailableModels is paginated via a top-level "nextToken". Both caps are hard
# safety limits: a broken or hostile upstream that always returns a nextToken (or an
# unbounded catalog) must not spin forever or exhaust memory. 19 models fit on one page
# today, so in practice neither cap is reached.
MODEL_DISCOVERY_MAX_PAGES: int = 10
MODEL_DISCOVERY_MAX_MODELS: int = 500


def _is_runtime_endpoint(auth_manager: KiroAuthManager) -> bool:
    """
    Check if auth manager uses the runtime (streaming) endpoint.
    
    Runtime endpoint pattern: https://runtime.{region}.kiro.dev
    Control-plane pattern:    https://q.{region}.amazonaws.com
    
    The runtime endpoint does not route /ListAvailableModels (every variant answers
    UnknownOperationException). Model LISTING for such accounts is therefore issued
    against the control-plane host derived by _get_model_listing_host(); chat and
    streaming keep using the runtime host unchanged.
    
    Args:
        auth_manager: KiroAuthManager instance
    
    Returns:
        True if using runtime endpoint, False otherwise
    
    Examples:
        >>> auth_manager.api_host = "https://runtime.us-east-1.kiro.dev"
        >>> _is_runtime_endpoint(auth_manager)
        True
        >>> auth_manager.api_host = "https://q.us-east-1.amazonaws.com"
        >>> _is_runtime_endpoint(auth_manager)
        False
    """
    return "://runtime." in auth_manager.api_host


# Extracts the region out of a runtime host, e.g. https://runtime.eu-central-1.kiro.dev
_RUNTIME_HOST_REGION_PATTERN = re.compile(r"://runtime\.(?P<region>[^./]+)\.kiro\.dev")


def _get_api_region(auth_manager: KiroAuthManager) -> str:
    """
    Best-effort recovery of the account's effective API region.
    
    KiroAuthManager exposes the resolved hosts but not the resolved API region, so the
    region is read back out of q_host/api_host (which were built from it) and only then
    falls back to the account's configured region.
    
    Args:
        auth_manager: KiroAuthManager instance
    
    Returns:
        Region string, e.g. "us-east-1"
    """
    for host in (auth_manager.q_host, auth_manager.api_host):
        match = _RUNTIME_HOST_REGION_PATTERN.search(host or "")
        if match:
            return match.group("region")
    return auth_manager.region or REGION


def _get_model_listing_host(auth_manager: KiroAuthManager) -> str:
    """
    Return the host that serves /ListAvailableModels for this account.
    
    For runtime-host accounts the catalog lives on the control plane
    (https://q.{api_region}.amazonaws.com) even though chat stays on
    runtime.{region}.kiro.dev. Legacy accounts already point q_host at a host that
    serves the operation, so it is used as-is.
    
    Args:
        auth_manager: KiroAuthManager instance
    
    Returns:
        Base host URL without a trailing slash
    """
    if _is_runtime_endpoint(auth_manager):
        return get_kiro_control_plane_host(_get_api_region(auth_manager))
    return auth_manager.q_host


# Maximum length of an upstream error body kept in a diagnostic reason
MAX_ERROR_BODY_CHARS = 300

# Credential fields that must never appear in logs or error messages
_SECRET_FIELD_NAMES = (
    "accessToken", "access_token",
    "refreshToken", "refresh_token",
    "idToken", "id_token",
    "clientSecret", "client_secret",
    "authorization", "Authorization",
    "bearer", "Bearer",
)

# "accessToken": "abc"  /  access_token=abc  /  Authorization: Bearer abc
_SECRET_PATTERN = re.compile(
    r'(?P<key>["\']?(?:' + "|".join(_SECRET_FIELD_NAMES) + r')["\']?\s*[:=]\s*)'
    r'(?P<quote>["\']?)(?P<value>[^"\'\s,;}&]+)(?P=quote)',
    re.IGNORECASE,
)

# Authorization: Bearer <token>  /  Bearer <token>
_BEARER_PATTERN = re.compile(
    r'(?P<prefix>(?:Authorization\s*[:=]\s*)?Bearer\s+)\S+',
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    """
    Remove credential values from a diagnostic string.
    
    Replaces the value of any known secret field (tokens, client secrets,
    Authorization headers) with "[REDACTED]" so error reasons can be logged
    safely.
    
    Args:
        text: Arbitrary diagnostic text (exception message, response body)
    
    Returns:
        Same text with secret values replaced
    
    Examples:
        >>> redact_secrets('{"refresh_token": "s3cret"}')
        '{"refresh_token": "[REDACTED]"}'
        >>> redact_secrets('Authorization: Bearer abc123')
        'Authorization: [REDACTED]'
    """
    if not text:
        return ""
    
    def _replace(match: "re.Match") -> str:
        quote = match.group("quote")
        return f'{match.group("key")}{quote}[REDACTED]{quote}'
    
    # Two passes: "Authorization: Bearer abc" needs both the header name and
    # the "Bearer" prefix handled.
    text = _BEARER_PATTERN.sub(lambda m: f'{m.group("prefix")}[REDACTED]', text)
    return _SECRET_PATTERN.sub(_replace, _SECRET_PATTERN.sub(_replace, text))


def describe_init_failure(exc: BaseException) -> str:
    """
    Build a safe, human-readable reason string for an initialization failure.
    
    Includes the exception type and message, plus the upstream response body
    for HTTP errors (truncated and with secrets redacted).
    
    Args:
        exc: Exception raised while initializing an account
    
    Returns:
        Reason string, e.g.
        'HTTPStatusError: HTTP 400 from AWS SSO OIDC: {"error":"invalid_grant"}'
    """
    reason = f"{type(exc).__name__}: {exc}"
    
    # Attach the upstream body for HTTP errors (httpx.HTTPStatusError and
    # anything else exposing a .response with .text)
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    if isinstance(body, str) and body.strip():
        status = getattr(response, "status_code", "?")
        reason = f"{type(exc).__name__}: HTTP {status}: {body}"
    
    reason = redact_secrets(reason).replace("\n", " ").strip()
    
    if len(reason) > MAX_ERROR_BODY_CHARS:
        reason = reason[:MAX_ERROR_BODY_CHARS] + "... (truncated)"
    
    return reason


def is_invalid_grant(reason: Optional[str]) -> bool:
    """
    Check whether a failure reason is an OIDC invalid_grant (revoked/expired token).
    
    Args:
        reason: Reason string from describe_init_failure()
    
    Returns:
        True if the upstream rejected the refresh token itself
    """
    return bool(reason) and isinstance(reason, str) and "invalid_grant" in reason.lower()


def format_init_failure_guidance(reason: Optional[str]) -> str:
    """
    Append actionable operator guidance to a failure reason.
    
    An invalid_grant means the stored refresh token was revoked upstream — the
    only fix is re-authenticating, not editing configuration.
    
    Args:
        reason: Reason string from describe_init_failure()
    
    Returns:
        Reason with guidance appended when applicable
    """
    if is_invalid_grant(reason):
        return (
            f"{reason} — the stored refresh token is no longer accepted by AWS SSO OIDC "
            f"and must be re-authenticated: log in again in Kiro IDE or kiro-cli, then "
            f"restart the gateway. This is not a configuration error."
        )
    return reason if isinstance(reason, str) and reason else "unknown error"


def _format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "30s", "5m", "2h", "1d")
    
    Examples:
        >>> _format_duration(30)
        '30s'
        >>> _format_duration(300)
        '5m'
        >>> _format_duration(7200)
        '2h'
        >>> _format_duration(86400)
        '1d'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h"
    else:
        return f"{int(seconds / 86400)}d"


@dataclass
class AccountStats:
    """
    Statistics for account usage.
    
    Tracks request counts for monitoring and future web UI.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


@dataclass
class Account:
    """
    Complete account entity with all dependencies.
    
    Represents a single Kiro account with its authentication,
    model cache, resolver, and runtime state.
    
    Attributes:
        id: Unique identifier (path to credentials file)
        auth_manager: Authentication manager (lazy initialized)
        model_cache: Model metadata cache (lazy initialized)
        model_resolver: Model resolver (lazy initialized)
        failures: Consecutive failure count (for Circuit Breaker)
        last_failure_time: Timestamp of last failure
        models_cached_at: Timestamp of last model cache update
        stats: Usage statistics
    """
    id: str
    auth_manager: Optional[KiroAuthManager] = None
    model_cache: Optional[ModelInfoCache] = None
    model_resolver: Optional[ModelResolver] = None
    failures: int = 0
    last_failure_time: float = 0.0
    models_cached_at: float = 0.0
    stats: AccountStats = field(default_factory=AccountStats)


@dataclass
class ModelAccountList:
    """
    List of accounts for a specific model.
    
    Attributes:
        accounts: List of account IDs that have this model
    
    Note: next_index removed - now using global _current_account_index
    """
    accounts: List[str] = field(default_factory=list)


class AccountManager:
    """
    Manages multiple Kiro accounts with intelligent failover.
    
    Responsibilities:
    - Load credentials from credentials.json
    - Lazy initialization of accounts
    - Select next available account (Circuit Breaker + Sticky)
    - Track statistics and failures
    - Persist state to state.json
    
    Example:
        >>> manager = AccountManager("credentials.json", "state.json")
        >>> await manager.load_credentials()
        >>> await manager.load_state()
        >>> account = await manager.get_next_account("claude-opus-4.5")
        >>> await manager.report_success(account.id, "claude-opus-4.5")
    """
    
    def __init__(self, credentials_file: str, state_file: str):
        """
        Initialize AccountManager.
        
        Args:
            credentials_file: Path to credentials.json
            state_file: Path to state.json
        """
        self._credentials_file = credentials_file
        self._state_file = state_file
        self._accounts: Dict[str, Account] = {}
        self._model_to_accounts: Dict[str, ModelAccountList] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._credentials_config: List[Dict] = []
        self._current_account_index: int = 0  # GLOBAL sticky index for all models
        # Last initialization failure reason per account (diagnostics only,
        # never contains credential values). See describe_init_failure().
        self._init_errors: Dict[str, str] = {}
        # Accounts for which a model-discovery failure has already been logged at
        # WARNING. Discovery failure is expected and non-fatal, so it is announced
        # once per account instead of on every TTL cycle.
        self._discovery_warned: set = set()
    
    def get_init_error(self, account_id: str) -> Optional[str]:
        """
        Get the reason the last initialization attempt for an account failed.
        
        Args:
            account_id: Account ID
        
        Returns:
            Redacted reason string, or None if the account never failed
        """
        return self._init_errors.get(account_id)
    
    async def load_credentials(self) -> None:
        """
        Load credentials from credentials.json.
        
        Validates each entry and creates Account objects.
        Invalid entries are skipped with warnings.
        Folders are scanned for credential files.
        """
        creds_path = Path(self._credentials_file).expanduser()
        
        if not creds_path.exists():
            logger.warning(f"Credentials file not found: {self._credentials_file}")
            return
        
        try:
            with open(creds_path, 'r', encoding='utf-8') as f:
                self._credentials_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return
        
        # Process each credential entry
        for entry in self._credentials_config:
            cred_type = entry.get("type")
            path = entry.get("path")
            enabled = entry.get("enabled", True)
            
            if not enabled:
                continue
            
            # Validate required fields based on type
            if not cred_type:
                logger.warning(f"Invalid credential entry (missing type): {entry}")
                continue
            
            # For json/sqlite types, path is required
            if cred_type in ("json", "sqlite") and not path:
                logger.warning(f"Invalid credential entry (type={cred_type} requires path): {entry}")
                continue
            
            # For refresh_token type, refresh_token field is required
            if cred_type == "refresh_token" and not entry.get("refresh_token"):
                logger.warning(f"Invalid credential entry (type=refresh_token requires refresh_token field): {entry}")
                continue
            
            # Handle refresh_token type (no path processing needed)
            if cred_type == "refresh_token":
                # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                token = entry.get('refresh_token', '')
                token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                account_id = f"refresh_token_{token_hash}"
                self._accounts[account_id] = Account(id=account_id)
                logger.debug(f"Added account: {account_id}")
                continue  # Skip path processing for refresh_token
            
            # Handle folder scanning for json/sqlite types
            expanded_path = Path(path).expanduser()
            if expanded_path.is_dir():
                logger.info(f"Scanning folder for credentials: {path}")
                for file_path in expanded_path.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    # Validate file before adding as account
                    account_id = str(file_path.resolve())
                    is_valid = False
                    
                    # Try JSON validation
                    if cred_type == "json":
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # Valid if has refreshToken or clientId
                                if 'refreshToken' in data or 'clientId' in data:
                                    is_valid = True
                        except Exception as e:
                            logger.warning(f"Invalid JSON credentials file {file_path.name}: {e}")
                    
                    # Try SQLite validation
                    elif cred_type == "sqlite":
                        try:
                            import sqlite3
                            conn = sqlite3.connect(str(file_path))
                            cursor = conn.cursor()
                            # Check if auth_kv table exists
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_kv'")
                            if cursor.fetchone():
                                is_valid = True
                            conn.close()
                        except Exception as e:
                            logger.warning(f"Invalid SQLite database file {file_path.name}: {e}")
                    
                    if is_valid:
                        self._accounts[account_id] = Account(id=account_id)
                        logger.debug(f"Added account from folder: {account_id}")
                    else:
                        logger.warning(f"Skipping invalid credentials file: {file_path.name}")
            elif expanded_path.is_file() or cred_type == "refresh_token":
                # Single file or refresh_token type
                if cred_type == "refresh_token":
                    # Use deterministic hash for refresh_token (hash() is not deterministic between process restarts)
                    token = entry.get('refresh_token', '')
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    account_id = f"refresh_token_{token_hash}"
                else:
                    account_id = str(expanded_path.resolve())
                self._accounts[account_id] = Account(id=account_id)
                logger.debug(f"Added account: {account_id}")
            else:
                logger.warning(f"Credential path not found: {path}")
        
        logger.info(f"Loaded {len(self._accounts)} account(s) from credentials")
    
    async def load_state(self) -> None:
        """
        Load runtime state from state.json.
        
        Restores model_to_accounts mapping and account runtime state.
        Creates empty state if file doesn't exist.
        """
        state_path = Path(self._state_file)
        
        if not state_path.exists():
            logger.debug("State file not found, starting with empty state")
            return
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            # Restore global current_account_index
            self._current_account_index = state_data.get("current_account_index", 0)
            
            # Restore model_to_accounts mapping (without next_index)
            for model, data in state_data.get("model_to_accounts", {}).items():
                self._model_to_accounts[model] = ModelAccountList(
                    accounts=data.get("accounts", [])
                )
            
            # Restore account runtime state
            for account_id, data in state_data.get("accounts", {}).items():
                if account_id in self._accounts:
                    account = self._accounts[account_id]
                    account.failures = data.get("failures", 0)
                    account.last_failure_time = data.get("last_failure_time", 0.0)
                    account.models_cached_at = data.get("models_cached_at", 0.0)
                    
                    stats_data = data.get("stats", {})
                    account.stats = AccountStats(
                        total_requests=stats_data.get("total_requests", 0),
                        successful_requests=stats_data.get("successful_requests", 0),
                        failed_requests=stats_data.get("failed_requests", 0)
                    )
            
            logger.info(f"Loaded state: {len(self._model_to_accounts)} model mappings, {len(self._accounts)} accounts")
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self) -> None:
        """
        Save runtime state to state.json atomically.
        
        Uses tmp file + rename for atomic write.
        """
        state_data = {
            "current_account_index": self._current_account_index,
            "accounts": {
                account_id: {
                    "failures": account.failures,
                    "last_failure_time": account.last_failure_time,
                    "models_cached_at": account.models_cached_at,
                    "stats": {
                        "total_requests": account.stats.total_requests,
                        "successful_requests": account.stats.successful_requests,
                        "failed_requests": account.stats.failed_requests
                    }
                }
                for account_id, account in self._accounts.items()
            },
            "model_to_accounts": {
                model: {
                    "accounts": mal.accounts
                }
                for model, mal in self._model_to_accounts.items()
            }
        }
        
        state_path = Path(self._state_file)
        tmp_path = state_path.with_suffix('.json.tmp')
        
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            tmp_path.replace(state_path)
            logger.debug("State saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    async def save_state_periodically(self) -> None:
        """
        Background task for periodic state saving.
        
        Saves state every STATE_SAVE_INTERVAL_SECONDS if dirty flag is set.
        """
        while True:
            await asyncio.sleep(STATE_SAVE_INTERVAL_SECONDS)
            
            if self._dirty:
                async with self._lock:
                    await self._save_state()
                    self._dirty = False
    
    def _discovery_failed(self, account_id: str, reason: str) -> None:
        """
        Record a non-fatal model-discovery failure and return None.
        
        Logged once per account at WARNING; later failures for the same account are
        DEBUG so a persistent control-plane outage cannot flood the log.
        
        Args:
            account_id: Account ID
            reason: Redacted, human-readable failure reason
        
        Returns:
            None, so callers can `return self._discovery_failed(...)`
        """
        message = (
            f"Model discovery failed for {account_id}: {reason}. "
            f"Falling back to the static model list ({len(FALLBACK_MODELS)} models); "
            f"discovery is retried after MODEL_CACHE_TTL."
        )
        if account_id in self._discovery_warned:
            logger.debug(message)
        else:
            self._discovery_warned.add(account_id)
            logger.warning(message)
        return None
    
    async def _discover_models(
        self,
        auth_manager: KiroAuthManager,
        account_id: str,
    ) -> Optional[List[Dict]]:
        """
        Fetch the live model catalog via /ListAvailableModels.
        
        Listing is decoupled from streaming: for runtime-host accounts the request goes
        to the control-plane host (see _get_model_listing_host()) with the same bearer
        token and get_kiro_headers() as everything else, while chat keeps using the
        runtime host.
        
        This method NEVER raises and NEVER blocks longer than MODEL_DISCOVERY_TIMEOUT,
        so neither startup nor request handling can be held up or broken by a
        control-plane outage.
        
        Args:
            auth_manager: Initialized auth manager for the account
            account_id: Account ID (diagnostics only)
        
        Returns:
            Non-empty list of model dicts (full upstream metadata preserved), or None
            when discovery is disabled or failed — callers then use FALLBACK_MODELS.
        """
        if not MODEL_DISCOVERY:
            logger.debug(
                f"Account {account_id}: model discovery disabled (MODEL_DISCOVERY=false), "
                f"using static model list"
            )
            return None
        
        params = {"origin": "AI_EDITOR"}
        if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
            params["profileArn"] = auth_manager.profile_arn
        
        list_models_url = f"{_get_model_listing_host(auth_manager)}/ListAvailableModels"
        
        http_client = KiroHttpClient(auth_manager, shared_client=None)
        
        async def _fetch_all_pages() -> tuple:
            """
            Follow nextToken across pages, returning (models, failure_reason).
            
            Page 1 uses exactly the same params as before (no extra keys), so the
            request shape for single-page accounts is unchanged. Subsequent pages add
            "nextToken". Bounded by MODEL_DISCOVERY_MAX_PAGES and
            MODEL_DISCOVERY_MAX_MODELS. A failure on page 1 is fatal (caller falls back);
            a failure on a later page keeps whatever was already collected.
            """
            collected: List[Dict] = []
            seen_ids = set()
            next_token = None
            
            for page in range(1, MODEL_DISCOVERY_MAX_PAGES + 1):
                page_params = dict(params)
                if next_token:
                    page_params["nextToken"] = next_token
                
                response = await http_client.request_with_retry(
                    method="GET",
                    url=list_models_url,
                    json_data=None,
                    params=page_params,
                    stream=False,
                )
                
                status = getattr(response, "status_code", None)
                if status != 200:
                    return collected, f"HTTP {status} from {list_models_url} (page {page})"
                
                data = response.json()
                if not isinstance(data, dict):
                    return collected, f"response body is not a JSON object (page {page})"
                
                raw_models = data.get("models")
                if not isinstance(raw_models, list) or not raw_models:
                    return collected, f"response contained no usable 'models' array (page {page})"
                
                for model in raw_models:
                    if not isinstance(model, dict):
                        continue
                    model_id = model.get("modelId")
                    if not isinstance(model_id, str) or not model_id or model_id in seen_ids:
                        continue
                    seen_ids.add(model_id)
                    collected.append(model)
                
                if len(collected) >= MODEL_DISCOVERY_MAX_MODELS:
                    logger.warning(
                        f"Account {account_id}: model discovery hit the "
                        f"{MODEL_DISCOVERY_MAX_MODELS}-model cap; ignoring further pages."
                    )
                    return collected, None
                
                next_token = data.get("nextToken")
                if not isinstance(next_token, str) or not next_token:
                    return collected, None
                
                if page == MODEL_DISCOVERY_MAX_PAGES:
                    logger.warning(
                        f"Account {account_id}: model discovery hit the "
                        f"{MODEL_DISCOVERY_MAX_PAGES}-page cap; ignoring further pages."
                    )
            
            return collected, None
        
        try:
            models, failure_reason = await asyncio.wait_for(
                _fetch_all_pages(),
                timeout=MODEL_DISCOVERY_TIMEOUT,
            )
            
            if not models:
                return self._discovery_failed(
                    account_id,
                    failure_reason or "no entry in 'models' had a usable modelId",
                )
            
            if failure_reason:
                logger.warning(
                    f"Account {account_id}: model discovery stopped early "
                    f"({failure_reason}); keeping the {len(models)} model(s) collected."
                )
            
            logger.info(
                f"Account {account_id}: discovered {len(models)} model(s) from {list_models_url}"
            )
            self._discovery_warned.discard(account_id)
            return models
        
        except asyncio.TimeoutError:
            return self._discovery_failed(
                account_id, f"timed out after {MODEL_DISCOVERY_TIMEOUT}s"
            )
        except Exception as e:
            return self._discovery_failed(account_id, describe_init_failure(e))
        finally:
            try:
                await http_client.close()
            except Exception as e:
                logger.debug(f"Error closing discovery HTTP client: {e}")
    
    async def _initialize_account(self, account_id: str) -> bool:
        """
        Initialize account (lazy initialization).
        
        Creates auth_manager, fetches models, creates cache and resolver.
        
        Args:
            account_id: Account ID to initialize
        
        Returns:
            True if successful, False otherwise
        
        Note:
            On failure the reason is recorded and available via
            get_init_error(account_id) — the boolean contract is unchanged.
        """
        account = self._accounts.get(account_id)
        if not account:
            self._init_errors[account_id] = "Unknown account (not present in credentials)"
            return False
        
        try:
            # Find credentials config for this account
            creds_config = None
            for entry in self._credentials_config:
                path = entry.get("path", "")
                expanded_path = Path(path).expanduser()
                
                if entry.get("type") == "refresh_token":
                    # Match by deterministic hash for refresh_token type
                    token = entry.get('refresh_token', '')
                    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
                    if account_id == f"refresh_token_{token_hash}":
                        creds_config = entry
                        break
                elif str(expanded_path.resolve()) == account_id or (expanded_path.is_dir() and account_id.startswith(str(expanded_path.resolve()) + os.sep)):
                    creds_config = entry
                    break
            
            if not creds_config:
                logger.error(f"No credentials config found for account: {account_id}")
                self._init_errors[account_id] = "No matching entry in credentials.json"
                return False
            
            # Create KiroAuthManager based on type
            cred_type = creds_config.get("type")
            if cred_type == "json":
                auth_manager = KiroAuthManager(
                    creds_file=account_id,
                    profile_arn=creds_config.get("profile_arn"),
                    region=creds_config.get("region", "us-east-1"),
                    api_region=creds_config.get("api_region")
                )
            elif cred_type == "sqlite":
                auth_manager = KiroAuthManager(
                    sqlite_db=account_id,
                    profile_arn=creds_config.get("profile_arn"),
                    region=creds_config.get("region", "us-east-1"),
                    api_region=creds_config.get("api_region")
                )
            elif cred_type == "refresh_token":
                auth_manager = KiroAuthManager(
                    refresh_token=creds_config.get("refresh_token"),
                    profile_arn=creds_config.get("profile_arn"),
                    region=creds_config.get("region", "us-east-1"),
                    api_region=creds_config.get("api_region")
                )
            else:
                logger.error(f"Unknown credential type: {cred_type}")
                self._init_errors[account_id] = f"Unknown credential type: {cred_type}"
                return False
            
            # Get token to verify credentials
            token = await auth_manager.get_access_token()
            
            # Determine model catalog: prefer the live upstream list, fall back to static.
            # Discovery never raises and is bounded by MODEL_DISCOVERY_TIMEOUT, so a
            # control-plane outage cannot fail or delay startup.
            discovered = await self._discover_models(auth_manager, account_id)
            models_list = discovered if discovered else FALLBACK_MODELS
            
            # Create model cache and update
            model_cache = ModelInfoCache()
            await model_cache.update(models_list)
            
            # Add hidden models
            for display_name, internal_id in HIDDEN_MODELS.items():
                model_cache.add_hidden_model(display_name, internal_id)
            
            # Create model resolver
            model_resolver = ModelResolver(
                cache=model_cache,
                hidden_models=HIDDEN_MODELS,
                aliases=MODEL_ALIASES,
                hidden_from_list=HIDDEN_FROM_LIST
            )
            
            # Update account
            account.auth_manager = auth_manager
            account.model_cache = model_cache
            account.model_resolver = model_resolver
            account.models_cached_at = time.time()
            
            # Update model_to_accounts mapping
            available_models = model_resolver.get_available_models()
            for model in available_models:
                if model not in self._model_to_accounts:
                    self._model_to_accounts[model] = ModelAccountList()
                if account_id not in self._model_to_accounts[model].accounts:
                    self._model_to_accounts[model].accounts.append(account_id)
            
            logger.info(f"Initialized account: {account_id} ({len(available_models)} models)")
            self._dirty = True
            self._init_errors.pop(account_id, None)
            return True
        
        except Exception as e:
            reason = describe_init_failure(e)
            self._init_errors[account_id] = reason
            logger.error(f"Failed to initialize account {account_id}: {reason}")
            return False
    
    async def _refresh_account_models(self, account_id: str, force: bool = False) -> None:
        """
        Refresh the model cache for an account (TTL refresh).
        
        Discovery is re-issued only when the cached catalog is older than
        MODEL_CACHE_TTL, so this is never a per-request fetch. On any failure the
        existing cache is kept (or FALLBACK_MODELS installed if there is none) and no
        exception escapes.
        
        Args:
            account_id: Account ID to refresh
            force: Ignore the TTL window and refresh now
        """
        account = self._accounts.get(account_id)
        if not account or not account.auth_manager or not account.model_cache:
            return
        
        # TTL gate: inside the window the cached catalog is reused as-is.
        if not force and account.models_cached_at > 0:
            age = time.time() - account.models_cached_at
            if age < MODEL_CACHE_TTL:
                logger.debug(
                    f"Account {account_id}: model cache is {int(age)}s old "
                    f"(< MODEL_CACHE_TTL={MODEL_CACHE_TTL}s), skipping refresh"
                )
                return
        
        discovered = await self._discover_models(account.auth_manager, account_id)
        
        if not discovered:
            # Discovery disabled or failed. Keep whatever we already serve; only install
            # the static list when the cache is empty, so chat never loses its catalog.
            if account.model_cache.is_empty():
                await account.model_cache.update(FALLBACK_MODELS)
                for display_name, internal_id in HIDDEN_MODELS.items():
                    account.model_cache.add_hidden_model(display_name, internal_id)
            account.models_cached_at = time.time()
            self._dirty = True
            return
        
        # Live list wins outright (matching previous dynamic behaviour).
        await account.model_cache.update(discovered)
        for display_name, internal_id in HIDDEN_MODELS.items():
            account.model_cache.add_hidden_model(display_name, internal_id)
        account.models_cached_at = time.time()
        
        # Update model_to_accounts mapping (new models may have appeared)
        if account.model_resolver:
            available_models = account.model_resolver.get_available_models()
            for model in available_models:
                if model not in self._model_to_accounts:
                    self._model_to_accounts[model] = ModelAccountList()
                if account_id not in self._model_to_accounts[model].accounts:
                    self._model_to_accounts[model].accounts.append(account_id)
        
        logger.debug(f"Refreshed models for {account_id}")
        self._dirty = True
    
    async def get_next_account(self, model: str, exclude_accounts: Optional[set] = None) -> Optional[Account]:
        """
        Get next available account for model (Circuit Breaker + Sticky).
        
        Implements:
        - Sticky behavior (prefer successful account)
        - Circuit Breaker with exponential backoff
        - Probabilistic retry for "dead" accounts (10%)
        - TTL-based model cache refresh
        - Exclusion of already-tried accounts in current failover loop
        
        Args:
            model: Model name (will be normalized)
            exclude_accounts: Set of account IDs to exclude (already tried in current failover loop)
        
        Returns:
            Account object or None if no accounts available
        """
        async with self._lock:
            # Special case: single account - bypass Circuit Breaker
            # Circuit Breaker is meaningless for single account - user should see real Kiro API errors
            # instead of generic "Account unavailable" after cooldown kicks in
            if len(self._accounts) == 1:
                account_id = list(self._accounts.keys())[0]
                account = self._accounts[account_id]
                
                # Skip if already tried in current failover loop
                if exclude_accounts and account_id in exclude_accounts:
                    return None
                
                # Lazy initialization if needed
                if account.auth_manager is None:
                    success = await self._initialize_account(account_id)
                    if not success:
                        return None
                
                # Check TTL and refresh if needed
                if account.models_cached_at > 0:
                    age = time.time() - account.models_cached_at
                    # MODEL_CACHE_TTL drives catalog freshness; ACCOUNT_CACHE_TTL is
                    # kept as an upper bound for backward compatibility.
                    if age > min(ACCOUNT_CACHE_TTL, MODEL_CACHE_TTL):
                        try:
                            await self._refresh_account_models(account_id)
                        except Exception as e:
                            logger.warning(f"Failed to refresh models for {account_id}: {e}")
                # # Validate model availability
                # if account.model_resolver:
                #     normalized_model = normalize_model_name(model)
                #     available_models = account.model_resolver.get_available_models()
                #     if normalized_model not in available_models:
                #         return None
                
                # Always return single account (ignore cooldown/failures)
                # No model validation - let Kiro API decide (gateway, not gatekeeper)
                return account
            
            # Multi-account logic: GLOBAL sticky
            normalized_model = normalize_model_name(model)
            
            # ALWAYS start from GLOBAL index (one current account for ALL models)
            start_index = self._current_account_index
            
            # ALWAYS iterate over ALL accounts
            all_account_ids = list(self._accounts.keys())
            
            for i in range(len(all_account_ids)):
                current_index = (start_index + i) % len(all_account_ids)
                account_id = all_account_ids[current_index]
                account = self._accounts[account_id]
                
                # Skip accounts already tried in current failover loop
                if exclude_accounts and account_id in exclude_accounts:
                    continue
                
                # Check Circuit Breaker (Half-Open state with exponential backoff)
                if account.failures > 0:
                    time_since_failure = time.time() - account.last_failure_time
                    
                    # Exponential backoff: base * 2^(failures - 1), capped at MAX_MULTIPLIER
                    # 1 failure: 60s, 2: 120s, 3: 240s, ..., 12+: 86400s (1 day cap)
                    backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                    effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                    
                    if time_since_failure < effective_timeout:
                        # Probabilistic retry (10% chance)
                        if random.random() > ACCOUNT_PROBABILISTIC_RETRY_CHANCE:
                            continue
                        else:
                            logger.info(f"Probabilistic retry for broken account {account_id}")
                    else:
                        # Half-Open: recovery timeout passed
                        logger.info(f"Half-Open state for {account_id} (recovery timeout passed, effective={effective_timeout}s)")
                
                # Lazy initialization
                if account.auth_manager is None:
                    success = await self._initialize_account(account_id)
                    if not success:
                        account.failures += 1
                        self._dirty = True
                        continue
                
                # Check TTL and refresh if needed
                if account.models_cached_at > 0:
                    age = time.time() - account.models_cached_at
                    # MODEL_CACHE_TTL drives catalog freshness; ACCOUNT_CACHE_TTL is
                    # kept as an upper bound for backward compatibility.
                    if age > min(ACCOUNT_CACHE_TTL, MODEL_CACHE_TTL):
                        try:
                            await self._refresh_account_models(account_id)
                        except Exception as e:
                            logger.warning(f"Failed to refresh models for {account_id}: {e}")
                # # Check if model is available on this account
                # available_models = account.model_resolver.get_available_models()
                # if normalized_model not in available_models:
                #     continue
                
                # No model validation - let Kiro API decide (gateway, not gatekeeper)
                # Account is suitable!
                return account
            
            # All accounts unavailable
            return None
    
    async def report_success(self, account_id: str, model: str) -> None:
        """
        Report successful request (reset failures, update stats, sticky, dynamic learning).
        
        Args:
            account_id: Account ID
            model: Model name
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Reset failures
            if account.failures > 0:
                account.failures = 0
                self._dirty = True
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.successful_requests += 1
            self._dirty = True
            
            # Dynamic learning: add model to mapping if successful
            # This allows system to learn about new models not in FALLBACK_MODELS
            normalized_model = normalize_model_name(model)
            if normalized_model not in self._model_to_accounts:
                self._model_to_accounts[normalized_model] = ModelAccountList()
                logger.debug(f"Dynamic learning: discovered new model '{normalized_model}'")
            if account_id not in self._model_to_accounts[normalized_model].accounts:
                self._model_to_accounts[normalized_model].accounts.append(account_id)
                logger.debug(f"Dynamic learning: model '{normalized_model}' works on account {account_id}")
                self._dirty = True
            
            # GLOBAL STICKY: Update global current_account_index
            all_account_ids = list(self._accounts.keys())
            try:
                successful_index = all_account_ids.index(account_id)
                if self._current_account_index != successful_index:
                    self._current_account_index = successful_index
                    self._dirty = True
            except ValueError:
                pass
    
    async def report_failure(
        self,
        account_id: str,
        model: str,
        error_type: ErrorType,
        status_code: int,
        reason: Optional[str]
    ) -> None:
        """
        Report failed request (update failures, stats, failover).
        
        Args:
            account_id: Account ID
            model: Model name
            error_type: Error classification (FATAL or RECOVERABLE)
            status_code: HTTP status code
            reason: Error reason from Kiro API
        """
        async with self._lock:
            account = self._accounts.get(account_id)
            if not account:
                return
            
            # Special case: INVALID_MODEL_ID is discovery process, not account failure
            # Account is healthy, model is just not available on this account
            # Log for user visibility but don't penalize account statistics
            if reason == "INVALID_MODEL_ID":
                account.stats.total_requests += 1
                self._dirty = True
                logger.warning(
                    f"Model '{model}' not available on account {account_id}: "
                    f"status={status_code}, reason={reason}"
                )
                return
            
            # Update failure count (only for RECOVERABLE)
            if error_type == ErrorType.RECOVERABLE:
                account.failures += 1
                account.last_failure_time = time.time()
                self._dirty = True
                
                # Calculate backoff for logging
                backoff_multiplier = min(2 ** (account.failures - 1), ACCOUNT_MAX_BACKOFF_MULTIPLIER)
                effective_timeout = ACCOUNT_RECOVERY_TIMEOUT * backoff_multiplier
                logger.warning(
                    f"Account {account_id} failure #{account.failures}: "
                    f"status={status_code}, reason={reason}, "
                    f"cooldown={_format_duration(effective_timeout)}"
                )
            
            # Update stats
            account.stats.total_requests += 1
            account.stats.failed_requests += 1
            self._dirty = True
            
            # GLOBAL STICKY: Do NOT change _current_account_index on failure
            # It only changes on success (GLOBAL sticky behavior)
            # Failover happens through exclude_accounts in get_next_account()
    
    def get_first_account(self) -> Account:
        """
        Get first initialized account (for legacy mode).
        
        Returns:
            First initialized account
        
        Raises:
            RuntimeError: If no initialized accounts available
        """
        for account in self._accounts.values():
            if account.auth_manager is not None:
                return account
        raise RuntimeError("No initialized accounts available")
    
    def iter_initialized_accounts(self):
        """
        Yield every account whose auth_manager has been initialized.
        
        Public read-only accessor for introspection (e.g. building the /v1/models
        response) so callers do not reach into self._accounts directly.
        
        Yields:
            Account: initialized accounts, in insertion order.
        """
        for account in self._accounts.values():
            if account.auth_manager is not None:
                yield account
    
    def get_all_available_models(self) -> List[str]:
        """
        Collect unique models from all initialized accounts.
        
        Used by /v1/models endpoint in account system to show
        all available models across all accounts.
        
        Returns:
            Sorted list of unique model IDs
        """
        all_models = set()
        for account in self._accounts.values():
            if account.model_resolver:
                all_models.update(account.model_resolver.get_available_models())
        return sorted(all_models)
