"""Shared token-endpoint credential probe for IM channels.

This is the common building block for channels whose credentials validate
against a token endpoint (DingTalk's OAuth ``accessToken``, QQ's
``getAppAccessToken``). Everything here is stateless: callers pass the
endpoint URL, the JSON payload and the secret values explicitly, so a probe
never touches a channel's shared HTTP client and an unsaved credential set
can be checked before the channel is started.

Secrets are scrubbed from every ``detail`` before it reaches a result
envelope, and a successfully fetched token is discarded: it is never
returned, logged, or cached.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import httpx


def scrub_secrets(text: str, secrets: Iterable[str]) -> str:
    """Return diagnostic text with every non-empty secret value replaced.

    Transport and rejection messages can echo the request or credential by
    accident; each value the caller holds as a secret is masked before the
    text reaches a result envelope.
    """
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def response_detail(resp: httpx.Response) -> str:
    """Return a bounded ``HTTP <status>`` detail without echoing secrets."""
    body = (resp.text or "").strip()
    if not body:
        return f"HTTP {resp.status_code}"
    return f"HTTP {resp.status_code}: {body[:200]}"


async def probe_token_endpoint(
    *,
    url: str,
    payload: dict[str, Any],
    token_key: str,
    secrets: Sequence[str],
    sdk_available: bool,
    transport: httpx.AsyncHTTPTransport | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Validate credentials with a standalone token-endpoint request.

    Uses a fresh ``httpx.AsyncClient`` rather than any channel's shared
    client, so the probe works before (or without) the channel being started.
    A successful token is discarded: it is never returned, logged, or cached.

    Args:
        url: The token endpoint to POST to.
        payload: JSON body carrying the credentials.
        token_key: Response field whose truthy value means success.
        secrets: Credential values to scrub from every ``detail``.
        sdk_available: Whether the channel's optional SDK imported; echoed
            back so the caller's envelope carries it unchanged.
        transport: Optional custom transport (e.g. an IPv4-bound one).
        timeout: Per-phase timeout in seconds for the fresh client.

    Returns:
        A JSON-serializable envelope with ``ok`` and a ``code`` of ``ok`` /
        ``invalid_credentials`` / ``network``, plus an ``sdk_available``
        flag. Any ``detail`` is scrubbed of credential values.
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=timeout),
            transport=transport,
        ) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "code": "network",
            "detail": scrub_secrets(str(exc) or type(exc).__name__, secrets),
            "sdk_available": sdk_available,
        }

    if resp.status_code == 200:
        try:
            token = resp.json().get(token_key)
        except ValueError:
            token = None
        if not token:
            return {
                "ok": False,
                "code": "invalid_credentials",
                "detail": "no access token in response",
                "sdk_available": sdk_available,
            }
        return {"ok": True, "code": "ok", "sdk_available": sdk_available}

    if 400 <= resp.status_code < 500:
        return {
            "ok": False,
            "code": "invalid_credentials",
            "detail": scrub_secrets(response_detail(resp), secrets),
            "sdk_available": sdk_available,
        }

    return {
        "ok": False,
        "code": "network",
        "detail": scrub_secrets(response_detail(resp), secrets),
        "sdk_available": sdk_available,
    }
