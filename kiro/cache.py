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
Model metadata cache for Kiro Gateway.

Thread-safe storage for available model information
with TTL and lazy loading support.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import MODEL_CACHE_TTL, DEFAULT_MAX_INPUT_TOKENS, MODEL_ALIASES


class ModelInfoCache:
    """
    Thread-safe cache for storing model metadata.
    
    Uses Lazy Loading for population - data is loaded
    only on first access or when cache is stale.
    
    Attributes:
        cache_ttl: Cache time-to-live in seconds
    
    Example:
        >>> cache = ModelInfoCache()
        >>> await cache.update([{"modelId": "claude-sonnet-4", "tokenLimits": {...}}])
        >>> info = cache.get("claude-sonnet-4")
        >>> max_tokens = cache.get_max_input_tokens("claude-sonnet-4")
    """
    
    def __init__(self, cache_ttl: int = MODEL_CACHE_TTL):
        """
        Initializes the model cache.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default from config)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl
    
    # Richer per-model metadata returned by ListAvailableModels. Stored verbatim so a
    # later change can expose it without another upstream round-trip.
    METADATA_FIELDS = (
        "modelName",
        "description",
        "tokenLimits",
        "promptCaching",
        "rateMultiplier",
        "rateUnit",
        "supportedInputTypes",
        "availableOrigins",
        "additionalModelRequestFieldsSchema",
    )

    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """
        Updates the model cache.
        
        Thread-safely replaces cache contents with new data.
        
        Entries that are not dicts or carry no non-empty "modelId" are skipped instead
        of raising, so a malformed upstream body can never break discovery. Every other
        field of an accepted entry is preserved verbatim (see METADATA_FIELDS).
        
        Args:
            models_data: List of dictionaries with model information.
                        Each dictionary must contain the "modelId" key.
        """
        async with self._lock:
            normalized: Dict[str, Dict[str, Any]] = {}
            skipped = 0
            for model in models_data or []:
                if not isinstance(model, dict):
                    skipped += 1
                    continue
                model_id = model.get("modelId")
                if not isinstance(model_id, str) or not model_id:
                    skipped += 1
                    continue
                normalized[model_id] = model
            if skipped:
                logger.warning(f"Skipped {skipped} malformed model entrie(s) while updating cache.")
            logger.info(f"Updating model cache. Found {len(normalized)} models.")
            self._cache = normalized
            self._last_update = time.time()

    def get_metadata(self, model_id: str, field: str) -> Optional[Any]:
        """
        Return one metadata field for a model, or None when absent.
        
        Args:
            model_id: Model ID
            field: Metadata key (e.g. "rateMultiplier", "promptCaching")
        
        Returns:
            Field value or None
        """
        model = self._cache.get(model_id)
        if not model:
            return None
        return model.get(field)
    
    def get_public_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Return the upstream metadata for a model in public (snake_case) form.
        
        Only fields actually reported by ListAvailableModels are returned; anything
        absent, null or of an unexpected type is omitted rather than defaulted, so
        callers can never publish a fabricated limit. A model that carries no metadata
        at all (e.g. an entry from the static FALLBACK_MODELS list, which has only
        "modelId") yields an empty dict.
        
        Synthetic entries created by add_hidden_model() (public aliases such as
        "auto-kiro") carry no upstream metadata of their own — their tokenLimits are a
        local default, not an upstream measurement — so they INHERIT the metadata of the
        internal model they point at ("_internal_id"). Only metadata is inherited; the
        alias keeps its own public id, which this method never emits. When the internal
        model is itself unknown or metadata-free (e.g. a static FALLBACK_MODELS entry),
        the result is an empty dict, exactly as before.
        
        Args:
            model_id: Model ID
        
        Returns:
            Dict with any of: display_name, context_length, max_input_tokens,
            max_output_tokens, supported_input_types, supports_prompt_caching,
            max_cache_checkpoints, min_tokens_per_cache_checkpoint, rate_multiplier,
            rate_unit. Empty dict when nothing is known.
        """
        model = self._cache.get(model_id)
        if not model:
            return {}
        if model.get("_is_hidden"):
            internal_id = model.get("_internal_id")
            if not isinstance(internal_id, str) or not internal_id or internal_id == model_id:
                return {}
            target = self._cache.get(internal_id)
            # Only a real upstream entry may be inherited from; a chain of synthetic
            # entries would have nothing measured to offer.
            if not target or target.get("_is_hidden"):
                return {}
            return self._extract_public_metadata(target)
        
        return self._extract_public_metadata(model)
    
    @staticmethod
    def _extract_public_metadata(model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project one raw upstream model entry into public (snake_case) metadata.
        
        Anything absent, null or of an unexpected type is omitted rather than defaulted,
        so a caller can never publish a fabricated limit.
        """
        out: Dict[str, Any] = {}
        
        display_name = model.get("modelName")
        if isinstance(display_name, str) and display_name:
            out["display_name"] = display_name
        
        token_limits = model.get("tokenLimits")
        if isinstance(token_limits, dict):
            max_input = token_limits.get("maxInputTokens")
            max_output = token_limits.get("maxOutputTokens")
            if isinstance(max_input, int) and not isinstance(max_input, bool):
                # context_length mirrors max_input_tokens (naming precedent from
                # upstream PR #156); both are published so either convention works.
                out["context_length"] = max_input
                out["max_input_tokens"] = max_input
            if isinstance(max_output, int) and not isinstance(max_output, bool):
                out["max_output_tokens"] = max_output
        
        input_types = model.get("supportedInputTypes")
        if isinstance(input_types, list):
            cleaned = [t for t in input_types if isinstance(t, str) and t]
            if cleaned:
                out["supported_input_types"] = cleaned
        
        caching = model.get("promptCaching")
        if isinstance(caching, dict):
            supported = caching.get("supportsPromptCaching")
            if isinstance(supported, bool):
                out["supports_prompt_caching"] = supported
            checkpoints = caching.get("maximumCacheCheckpointsPerRequest")
            if isinstance(checkpoints, int) and not isinstance(checkpoints, bool):
                out["max_cache_checkpoints"] = checkpoints
            min_tokens = caching.get("minimumTokensPerCacheCheckpoint")
            if isinstance(min_tokens, int) and not isinstance(min_tokens, bool):
                out["min_tokens_per_cache_checkpoint"] = min_tokens
        
        rate_multiplier = model.get("rateMultiplier")
        if isinstance(rate_multiplier, (int, float)) and not isinstance(rate_multiplier, bool):
            out["rate_multiplier"] = rate_multiplier
        
        rate_unit = model.get("rateUnit")
        if isinstance(rate_unit, str) and rate_unit:
            out["rate_unit"] = rate_unit
        
        return out
    
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns model information.
        
        Args:
            model_id: Model ID
        
        Returns:
            Dictionary with model information or None if model not found
        """
        return self._cache.get(model_id)
    
    def is_valid_model(self, model_id: str) -> bool:
        """
        Check if model exists in dynamic cache.
        
        Used by ModelResolver to verify if a model is available.
        
        Args:
            model_id: Model ID to check
        
        Returns:
            True if model exists in cache, False otherwise
        """
        return model_id in self._cache
    
    def add_hidden_model(self, display_name: str, internal_id: str) -> None:
        """
        Add a hidden model to the cache.
        
        Hidden models are not returned by Kiro /ListAvailableModels API
        but are still functional. They are added to the cache so they
        appear in our /v1/models endpoint.
        
        Args:
            display_name: Model name to display (e.g., "claude-3.7-sonnet")
            internal_id: Internal Kiro ID (e.g., "CLAUDE_3_7_SONNET_20250219_V1_0")
        """
        if display_name not in self._cache:
            self._cache[display_name] = {
                "modelId": display_name,
                "modelName": display_name,
                "description": f"Hidden model (internal: {internal_id})",
                # No "tokenLimits": a synthetic entry has measured nothing, and a
                # fabricated limit here was indistinguishable from an upstream one
                # (it silently became the denominator for context-usage inversion).
                # The local default is kept under a clearly-marked private key so
                # callers can tell "unknown" from "measured".
                "_default_max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
                "_internal_id": internal_id,  # Store internal ID for reference
                "_is_hidden": True,  # Mark as hidden model
            }
            logger.debug(f"Added hidden model: {display_name} → {internal_id}")
    
    @staticmethod
    def _measured_max_input_tokens(entry: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        Return the UPSTREAM-MEASURED maxInputTokens of one raw entry, else None.
        
        A synthetic entry (``_is_hidden``) has measured nothing, so it never
        contributes a limit of its own; neither does an entry whose tokenLimits are
        absent, malformed, null or non-positive (e.g. a static FALLBACK_MODELS entry,
        which carries only "modelId").
        """
        if not isinstance(entry, dict) or entry.get("_is_hidden"):
            return None
        limits = entry.get("tokenLimits")
        if not isinstance(limits, dict):
            return None
        value = limits.get("maxInputTokens")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    def resolve_max_input_tokens(self, model_id: str) -> Optional[int]:
        """
        Return the real maxInputTokens for a model, or None when it is unknown.
        
        Unlike :meth:`get_max_input_tokens` this NEVER substitutes a default, so a
        caller that must not derive figures from a guessed denominator (context-usage
        inversion) can tell "measured" from "unknown".
        
        Resolution is the same inheritance idea already used for public metadata and is
        strictly SINGLE-HOP, so self-referential or cyclic configuration terminates:
        
        1. The model's own measured tokenLimits.
        2. A synthetic alias entry (``add_hidden_model``) -> the limits of the internal
           model it points at (``_internal_id``): "auto-kiro" yields "auto"'s limit.
        3. An id that is not a cache entry at all but IS a key of MODEL_ALIASES (how
           "auto-kiro" reaches the visible set) -> the limits of the alias target.
        
        Args:
            model_id: Model ID or public alias
        
        Returns:
            Measured maxInputTokens, or None when nothing is known.
        """
        entry = self._cache.get(model_id)
        
        direct = self._measured_max_input_tokens(entry)
        if direct is not None:
            return direct
        
        target: Optional[str] = None
        if isinstance(entry, dict) and entry.get("_is_hidden"):
            internal_id = entry.get("_internal_id")
            if isinstance(internal_id, str) and internal_id and internal_id != model_id:
                target = internal_id
        if target is None:
            # Read at call time so configuration (and tests) can change the map.
            alias_target = MODEL_ALIASES.get(model_id)
            if isinstance(alias_target, str) and alias_target and alias_target != model_id:
                target = alias_target
        if target is None:
            return None
        
        # Single hop only: the target is never itself resolved again.
        return self._measured_max_input_tokens(self._cache.get(target))

    def has_measured_max_input_tokens(self, model_id: str) -> bool:
        """True when a real (non-default) maxInputTokens is known for the model."""
        return self.resolve_max_input_tokens(model_id) is not None

    def get_max_input_tokens(self, model_id: str) -> int:
        """
        Returns maxInputTokens for the model, falling back to a local default.
        
        Always an int, for callers that need a usable number. Use
        :meth:`resolve_max_input_tokens` when a guessed value is not acceptable.
        
        Args:
            model_id: Model ID
        
        Returns:
            Maximum number of input tokens or DEFAULT_MAX_INPUT_TOKENS
        """
        resolved = self.resolve_max_input_tokens(model_id)
        if resolved is not None:
            return resolved
        entry = self._cache.get(model_id)
        if isinstance(entry, dict):
            local_default = entry.get("_default_max_input_tokens")
            if isinstance(local_default, int) and not isinstance(local_default, bool) and local_default > 0:
                return local_default
        return DEFAULT_MAX_INPUT_TOKENS
    
    def is_empty(self) -> bool:
        """
        Checks if the cache is empty.
        
        Returns:
            True if cache is empty
        """
        return not self._cache
    
    def is_stale(self) -> bool:
        """
        Checks if the cache is stale.
        
        Returns:
            True if cache is stale (more than cache_ttl seconds have passed)
            or if cache was never updated
        """
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl
    
    def get_all_model_ids(self) -> List[str]:
        """
        Returns a list of all model IDs in the cache.
        
        Returns:
            List of model IDs
        """
        return list(self._cache.keys())
    
    @property
    def size(self) -> int:
        """Number of models in the cache."""
        return len(self._cache)
    
    @property
    def last_update_time(self) -> Optional[float]:
        """Last update time (timestamp) or None."""
        return self._last_update