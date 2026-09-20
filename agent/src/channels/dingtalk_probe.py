"""Standalone DingTalk credential probe, split from ``dingtalk.py`` for size.

The probe validates an (possibly unsaved) credential set against the token
endpoint without touching the channel's shared HTTP client. Everything here
is stateless: functions take the ``DingTalkConfig`` explicitly so the module
never imports the adapter at runtime (only under ``TYPE_CHECKING``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from src.channels.dingtalk import DingTalkConfig

DINGTALK_ACCESS_TOKEN_URL = "https://api.dingtalk.com/v1.0/oauth2/accessToken"


def build_http_transport(config: DingTalkConfig) -> httpx.AsyncHTTPTransport | None:
    """Return an IPv4-bound transport when ``force_ipv4`` is set, else None.

    DingTalk's robot-send API enforces the app's egress-IP whitelist; on
    dual-stack networks the rotating IPv6 prefix keeps falling out of it, so
    operators can pin the egress to the (typically stable) IPv4 address.
    """
    if config.force_ipv4:
        return httpx.AsyncHTTPTransport(local_address="0.0.0.0")
    return None


def scrub_detail(config: DingTalkConfig, text: str) -> str:
    """Return diagnostic text with any credential value replaced.

    Transport and rejection messages can echo the request or credential by
    accident; every value the config holds as a secret is masked before the
    text reaches a result envelope.
    """
    cleaned = text
    for secret in (config.client_secret, config.client_id):
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def response_detail(resp: httpx.Response) -> str:
    """Return a bounded ``HTTP <status>`` detail without echoing secrets."""
    body = (resp.text or "").strip()
    if not body:
        return f"HTTP {resp.status_code}"
    return f"HTTP {resp.status_code}: {body[:200]}"


async def test_connection(
    config: DingTalkConfig, *, sdk_available: bool
) -> dict[str, Any]:
    """Validate the DingTalk credentials with a standalone token request.

    Uses a fresh ``httpx.AsyncClient`` rather than the channel's shared client
    (which only exists after :meth:`DingTalkChannel.start`), so an unsaved
    credential set can be checked before the channel is started. A successful
    access token is discarded: it is never returned, logged, or cached.

    Args:
        config: The DingTalk credential set to validate.
        sdk_available: Whether the optional ``dingtalk-stream`` SDK imported.

    Returns:
        A JSON-serializable envelope with ``ok`` and a ``code`` of ``ok`` /
        ``invalid_credentials`` / ``network``, plus an ``sdk_available``
        flag. Any ``detail`` is scrubbed of credential values.
    """
    if not config.client_id or not config.client_secret:
        return {
            "ok": False,
            "code": "invalid_credentials",
            "detail": "missing credentials",
        }

    payload = {"appKey": config.client_id, "appSecret": config.client_secret}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=10.0),
            transport=build_http_transport(config),
        ) as client:
            resp = await client.post(DINGTALK_ACCESS_TOKEN_URL, json=payload)
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "code": "network",
            "detail": scrub_detail(config, str(exc) or type(exc).__name__),
            "sdk_available": sdk_available,
        }

    if resp.status_code == 200:
        try:
            access_token = resp.json().get("accessToken")
        except ValueError:
            access_token = None
        if not access_token:
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
            "detail": scrub_detail(config, response_detail(resp)),
            "sdk_available": sdk_available,
        }

    return {
        "ok": False,
        "code": "network",
        "detail": scrub_detail(config, response_detail(resp)),
        "sdk_available": sdk_available,
    }
