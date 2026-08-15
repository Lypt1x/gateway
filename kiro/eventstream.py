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
Incremental decoder for the AWS ``vnd.amazon.eventstream`` wire format.

The upstream Kiro / CodeWhisperer streaming API returns a genuine AWS event
stream, exactly like the real Rust client decodes with
``aws-smithy-eventstream``. Each message on the wire is::

    +--------------------------------------------------------------+
    | total length      : u32 big-endian (whole frame, incl. CRCs) |
    | headers length    : u32 big-endian                           |
    | prelude CRC32     : u32 big-endian (over the first 8 bytes)  |
    +--------------------------------------------------------------+
    | headers           : `headers length` bytes                   |
    | payload           : total - headers - 16 bytes               |
    +--------------------------------------------------------------+
    | message CRC32     : u32 big-endian (over total-4 bytes)      |
    +--------------------------------------------------------------+

Each header is ``name_len:u8, name, value_type:u8, value``.

Design constraints this module deliberately honours:

- **Incremental.** A frame may be split across chunk boundaries at any byte.
  Bytes are buffered until a whole frame is present; nothing is decoded early.
- **No text decoding over binary.** Only header *string* values and the payload
  slice are ever decoded, and only once framing has been resolved. The raw
  byte stream is never passed through ``decode('utf-8', errors='ignore')``.
- **Bounded and non-looping.** Malformed, zero-length and absurdly large length
  fields are rejected rather than trusted, and a truncated tail simply stays in
  the buffer instead of spinning.
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = [
    "EventStreamError",
    "EventStreamFrame",
    "EventStreamDecoder",
    "PRELUDE_LENGTH",
    "MAX_FRAME_SIZE",
    "MAX_HEADERS_SIZE",
    "encode_frame",
]

# prelude = total_len(4) + headers_len(4) + prelude_crc(4)
PRELUDE_LENGTH = 12

# total_len(4) + headers_len(4) + prelude_crc(4) + message_crc(4)
_OVERHEAD = 16

# AWS caps event-stream messages at 16 MiB and headers at 128 KiB. Anything
# beyond that is corruption, not a big message, and must not be buffered for.
MAX_FRAME_SIZE = 16 * 1024 * 1024
MAX_HEADERS_SIZE = 128 * 1024

# Header value type ids (vnd.amazon.eventstream).
_HDR_BOOL_TRUE = 0
_HDR_BOOL_FALSE = 1
_HDR_BYTE = 2
_HDR_SHORT = 3
_HDR_INT = 4
_HDR_LONG = 5
_HDR_BYTE_ARRAY = 6
_HDR_STRING = 7
_HDR_TIMESTAMP = 8
_HDR_UUID = 9


class EventStreamError(Exception):
    """Raised when the byte stream is not valid event-stream framing.

    Callers are expected to treat this as "stop trusting the framing" and fall
    back to the legacy parser, never as a fatal request error.
    """


@dataclass
class EventStreamFrame:
    """One fully validated event-stream message.

    Attributes:
        headers: Decoded headers, e.g. ``{":event-type": "assistantResponseEvent"}``.
        payload: Raw payload bytes (still undecoded; usually JSON).
    """

    headers: Dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""

    @property
    def event_type(self) -> Optional[str]:
        """Value of ``:event-type``, or None."""
        value = self.headers.get(":event-type")
        return value if isinstance(value, str) else None

    @property
    def message_type(self) -> Optional[str]:
        """Value of ``:message-type`` (``event``, ``exception``, ``error``)."""
        value = self.headers.get(":message-type")
        return value if isinstance(value, str) else None

    @property
    def exception_type(self) -> Optional[str]:
        """Value of ``:exception-type``, present on in-band exception frames."""
        value = self.headers.get(":exception-type")
        return value if isinstance(value, str) else None

    @property
    def content_type(self) -> Optional[str]:
        """Value of ``:content-type``."""
        value = self.headers.get(":content-type")
        return value if isinstance(value, str) else None

    def payload_text(self) -> str:
        """Decode the payload as UTF-8, replacing (not deleting) bad bytes.

        ``replace`` rather than ``ignore``: framing has already established the
        exact payload boundaries, so an invalid sequence here is real corruption
        and must stay visible instead of being silently dropped.
        """
        return self.payload.decode("utf-8", errors="replace")


def _decode_headers(raw: bytes) -> Dict[str, Any]:
    """Decode the header block of one frame.

    Args:
        raw: Exactly ``headers length`` bytes.

    Returns:
        Mapping of header name to Python value.

    Raises:
        EventStreamError: On any malformed or out-of-bounds header.
    """
    headers: Dict[str, Any] = {}
    pos = 0
    size = len(raw)

    while pos < size:
        name_len = raw[pos]
        pos += 1
        if name_len == 0 or pos + name_len > size:
            raise EventStreamError("malformed header name length")
        try:
            name = raw[pos:pos + name_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EventStreamError(f"non-UTF-8 header name: {exc}") from exc
        pos += name_len

        if pos >= size:
            raise EventStreamError("header truncated before value type")
        value_type = raw[pos]
        pos += 1

        if value_type == _HDR_BOOL_TRUE:
            value: Any = True
        elif value_type == _HDR_BOOL_FALSE:
            value = False
        elif value_type in (_HDR_BYTE, _HDR_SHORT, _HDR_INT, _HDR_LONG, _HDR_TIMESTAMP):
            width = {
                _HDR_BYTE: 1,
                _HDR_SHORT: 2,
                _HDR_INT: 4,
                _HDR_LONG: 8,
                _HDR_TIMESTAMP: 8,
            }[value_type]
            if pos + width > size:
                raise EventStreamError("header integer truncated")
            fmt = {1: ">b", 2: ">h", 4: ">i", 8: ">q"}[width]
            value = struct.unpack(fmt, raw[pos:pos + width])[0]
            pos += width
        elif value_type == _HDR_UUID:
            if pos + 16 > size:
                raise EventStreamError("header uuid truncated")
            value = raw[pos:pos + 16]
            pos += 16
        elif value_type in (_HDR_BYTE_ARRAY, _HDR_STRING):
            if pos + 2 > size:
                raise EventStreamError("header value length truncated")
            value_len = struct.unpack(">H", raw[pos:pos + 2])[0]
            pos += 2
            if pos + value_len > size:
                raise EventStreamError("header value truncated")
            chunk = raw[pos:pos + value_len]
            pos += value_len
            if value_type == _HDR_STRING:
                try:
                    value = chunk.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise EventStreamError(f"non-UTF-8 header value: {exc}") from exc
            else:
                value = chunk
        else:
            raise EventStreamError(f"unknown header value type {value_type}")

        headers[name] = value

    return headers


class EventStreamDecoder:
    """Incremental, allocation-frugal event-stream frame decoder.

    Usage::

        decoder = EventStreamDecoder()
        decoder.feed(chunk)
        while True:
            frame = decoder.next_frame()   # None => need more bytes
            if frame is None:
                break
            handle(frame)

    ``next_frame`` raises :class:`EventStreamError` when the buffered bytes are
    demonstrably not event-stream framing (bad prelude CRC, bad message CRC,
    impossible lengths, malformed headers). It never raises merely because the
    buffer holds a partial frame.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.frames_decoded = 0

    # ---------------------------------------------------------------- buffer

    def feed(self, chunk: bytes) -> None:
        """Append raw bytes from the socket."""
        if chunk:
            self._buffer.extend(chunk)

    @property
    def buffered_bytes(self) -> int:
        """Number of undecoded bytes still held."""
        return len(self._buffer)

    def take_buffer(self) -> bytes:
        """Detach and return every undecoded byte (used when falling back)."""
        data = bytes(self._buffer)
        self._buffer.clear()
        return data

    def reset(self) -> None:
        """Drop all state."""
        self._buffer.clear()
        self.frames_decoded = 0

    # ------------------------------------------------------------ inspection

    def looks_like_eventstream(self) -> Optional[bool]:
        """Cheap, non-destructive framing probe on the buffered head.

        Returns:
            True if the head is a plausible, CRC-valid prelude; False if it
            definitely is not; None if fewer than ``PRELUDE_LENGTH`` bytes are
            buffered and the question cannot be answered yet.
        """
        if len(self._buffer) < PRELUDE_LENGTH:
            return None
        try:
            self._read_prelude()
        except EventStreamError:
            return False
        return True

    def _read_prelude(self) -> "tuple[int, int]":
        """Validate the prelude at the head of the buffer.

        Returns:
            ``(total_len, headers_len)``.

        Raises:
            EventStreamError: If the prelude is not self-consistent.
        """
        total_len, headers_len, prelude_crc = struct.unpack(
            ">III", self._buffer[:PRELUDE_LENGTH]
        )

        if zlib.crc32(bytes(self._buffer[:8])) & 0xFFFFFFFF != prelude_crc:
            raise EventStreamError("prelude CRC32 mismatch")

        # A frame with no headers and no payload is still 16 bytes. Anything
        # smaller, or a headers block that does not fit, is corruption.
        if total_len < _OVERHEAD or total_len > MAX_FRAME_SIZE:
            raise EventStreamError(f"implausible total length {total_len}")
        if headers_len > MAX_HEADERS_SIZE or headers_len > total_len - _OVERHEAD:
            raise EventStreamError(f"implausible headers length {headers_len}")

        return total_len, headers_len

    # -------------------------------------------------------------- decoding

    def next_frame(self) -> Optional[EventStreamFrame]:
        """Decode and consume one whole frame.

        Returns:
            The frame, or None when the buffer does not yet hold a complete one.

        Raises:
            EventStreamError: If the buffered bytes are not valid framing. The
            buffer is left untouched so the caller can hand it to a fallback.
        """
        if len(self._buffer) < PRELUDE_LENGTH:
            return None

        total_len, headers_len = self._read_prelude()

        if len(self._buffer) < total_len:
            # Partial frame: keep buffering. This is the normal chunk-boundary
            # case and must never be treated as an error.
            return None

        frame_bytes = bytes(self._buffer[:total_len])

        expected_crc = struct.unpack(">I", frame_bytes[total_len - 4:total_len])[0]
        if zlib.crc32(frame_bytes[:total_len - 4]) & 0xFFFFFFFF != expected_crc:
            raise EventStreamError("message CRC32 mismatch")

        headers_start = PRELUDE_LENGTH
        headers_end = headers_start + headers_len
        payload_end = total_len - 4

        headers = _decode_headers(frame_bytes[headers_start:headers_end])
        payload = frame_bytes[headers_end:payload_end]

        # Only consume once every check has passed, so a raised error leaves the
        # buffer intact for the fallback path.
        del self._buffer[:total_len]
        self.frames_decoded += 1

        return EventStreamFrame(headers=headers, payload=payload)


# ==================================================================================================
# Encoder (test/diagnostic helper)
# ==================================================================================================

def encode_frame(headers: Dict[str, Any], payload: bytes) -> bytes:
    """Build one valid event-stream frame. Used by tests and debug tooling.

    Only string, bool and int header values are supported, which covers every
    header the Kiro API is known to send (``:event-type``, ``:message-type``,
    ``:content-type``, ``:exception-type``).

    Args:
        headers: Header name to value mapping.
        payload: Raw payload bytes.

    Returns:
        The complete framed message.
    """
    header_bytes = bytearray()
    for name, value in headers.items():
        raw_name = name.encode("utf-8")
        if len(raw_name) > 255:
            raise ValueError(f"header name too long: {name}")
        header_bytes.append(len(raw_name))
        header_bytes.extend(raw_name)

        if isinstance(value, bool):
            header_bytes.append(_HDR_BOOL_TRUE if value else _HDR_BOOL_FALSE)
        elif isinstance(value, int):
            header_bytes.append(_HDR_INT)
            header_bytes.extend(struct.pack(">i", value))
        elif isinstance(value, str):
            raw_value = value.encode("utf-8")
            header_bytes.append(_HDR_STRING)
            header_bytes.extend(struct.pack(">H", len(raw_value)))
            header_bytes.extend(raw_value)
        elif isinstance(value, (bytes, bytearray)):
            header_bytes.append(_HDR_BYTE_ARRAY)
            header_bytes.extend(struct.pack(">H", len(value)))
            header_bytes.extend(value)
        else:
            raise ValueError(f"unsupported header value type: {type(value)!r}")

    headers_len = len(header_bytes)
    total_len = _OVERHEAD + headers_len + len(payload)

    prelude = struct.pack(">II", total_len, headers_len)
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)

    body = prelude + bytes(header_bytes) + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
