"""Tests for channel config field metadata and fail-safe secret masking.

Covers the uiHints registry (hand-written for DingTalk, derived elsewhere) and
the security acceptance criterion: no key matching :data:`SECRET_KEY_RE` ever
survives in the non-secret ``values`` half.
"""

from __future__ import annotations

import pytest

from src.channels.config_meta import (
    SECRET_KEY_RE,
    channel_field_hints,
    split_values_secrets,
)
from src.channels.registry import discover_channel_names, load_channel_class

# Helper modules that live beside the adapters but export no ``BaseChannel``:
# ``targets`` is a scheduled-delivery registry, ``config_meta`` is this module.
_NON_CHANNEL_MODULES = frozenset({"targets", "config_meta"})

# Every discovered adapter that is a real channel. The sweep below locks this
# surface at 16.
ADAPTER_NAMES: list[str] = sorted(
    name for name in discover_channel_names() if name not in _NON_CHANNEL_MODULES
)


def _dummy_value(value: object) -> object:
    """Return a non-empty stand-in for a ``default_config()`` value."""
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return ["dummy-list-item"]
    if isinstance(value, dict):
        return {"dummy": "dummy-value"}
    if isinstance(value, int):
        return 1234
    if isinstance(value, float):
        return 1.5
    return "dummy-secret-1234"


def _section_for(name: str) -> dict[str, object] | None:
    """Return ``default_config()`` populated with dummies, or None if unloadable."""
    try:
        config = load_channel_class(name).default_config()
    except Exception:  # noqa: BLE001 - adapters with missing SDKs must degrade
        return None
    if not isinstance(config, dict):
        return None
    return {key: _dummy_value(value) for key, value in config.items()}


# --- (a) hand-written DingTalk hints ---------------------------------------- #


def test_dingtalk_field_hints_exact_snapshot() -> None:
    """DingTalk hints match the frozen contract, with ``enabled`` excluded."""
    hints = channel_field_hints("dingtalk")
    assert hints == [
        {
            "key": "client_id",
            "type": "text",
            "secret": False,
            "required": True,
            "help_key": "settings.channels.fields.dingtalk.client_id",
        },
        {
            "key": "client_secret",
            "type": "password",
            "secret": True,
            "required": True,
            "help_key": "settings.channels.fields.dingtalk.client_secret",
        },
        {
            "key": "allow_from",
            "type": "list",
            "secret": False,
            "required": False,
            "help_key": "settings.channels.fields.dingtalk.allow_from",
        },
        {
            "key": "allow_remote_media_redirects",
            "type": "bool",
            "secret": False,
            "required": False,
            "help_key": "settings.channels.fields.dingtalk.allow_remote_media_redirects",
        },
        {
            "key": "remote_media_redirect_allowed_hosts",
            "type": "list",
            "secret": False,
            "required": False,
            "help_key": "settings.channels.fields.dingtalk.remote_media_redirect_allowed_hosts",
        },
        {
            "key": "group_user_isolation",
            "type": "bool",
            "secret": False,
            "required": False,
            "help_key": "settings.channels.fields.dingtalk.group_user_isolation",
        },
        {
            "key": "force_ipv4",
            "type": "bool",
            "secret": False,
            "required": False,
            "help_key": "settings.channels.fields.dingtalk.force_ipv4",
        },
    ]
    assert "enabled" not in {hint["key"] for hint in hints}


# --- (b) fallback derivation ------------------------------------------------- #


@pytest.mark.parametrize("name", ["discord", "websocket"])
def test_fallback_derives_types_and_secret_flags(name: str) -> None:
    """Adapters without hand-written hints derive metadata from default_config()."""
    config = _section_for(name)
    assert config is not None, f"{name} should expose default_config()"
    hints = channel_field_hints(name)
    assert hints, f"{name} fallback should derive at least one field"
    by_key = {hint["key"]: hint for hint in hints}
    assert "enabled" not in by_key
    assert any(SECRET_KEY_RE.search(key) for key in config)

    for key, value in config.items():
        if key == "enabled":
            continue
        hint = by_key[key]
        assert hint["required"] is False
        assert hint["help_key"] is None
        if SECRET_KEY_RE.search(key):
            assert hint["secret"] is True
            assert hint["type"] == "password"
        else:
            assert hint["secret"] is False
            if isinstance(value, bool):
                assert hint["type"] == "bool"
            elif isinstance(value, list):
                assert hint["type"] == "list"
            else:
                assert hint["type"] == "text"


def test_unloadable_adapter_is_handled_gracefully() -> None:
    """Adapters whose SDK is missing yield no hints instead of raising."""
    for name in ADAPTER_NAMES:
        try:
            load_channel_class(name)
        except Exception:  # noqa: BLE001 - this is the branch under test
            assert channel_field_hints(name) == []


# --- (c)/(d) fail-safe secret heuristic ------------------------------------ #


def test_unknown_channel_masks_secret_keys_and_strips_them_from_values() -> None:
    """A secret-shaped key is masked even without any hand-written hint."""
    values, secrets = split_values_secrets(
        "no_such_channel",
        {"webhook_secret": "abc12345", "bot_token": "x", "name": "n"},
    )
    assert values == {"name": "n"}
    assert secrets == {
        "webhook_secret": {"set": True, "masked": "****2345"},
        "bot_token": {"set": True, "masked": "****x"},
    }


def test_empty_secret_is_unset_and_unmasked() -> None:
    """An unset secret reports ``set: false`` and an empty mask."""
    values, secrets = split_values_secrets("no_such_channel", {"api_key": ""})
    assert values == {}
    assert secrets == {"api_key": {"set": False, "masked": ""}}


def test_non_secret_values_pass_through_verbatim() -> None:
    """Non-secret keys (including ``enabled``) are copied unchanged."""
    section = {"client_id": "abc", "allow_from": ["u1"], "enabled": True}
    values, secrets = split_values_secrets("dingtalk", section)
    assert values == section
    assert secrets == {}


def test_hint_marked_secret_is_masked_for_known_channel() -> None:
    """A hint-declared secret is masked and never reaches values."""
    values, secrets = split_values_secrets(
        "dingtalk", {"client_secret": "dummy-secret-1234"}
    )
    assert values == {}
    assert secrets == {"client_secret": {"set": True, "masked": "****1234"}}


def test_secret_key_regex_is_case_insensitive() -> None:
    assert SECRET_KEY_RE.search("API_KEY")
    assert SECRET_KEY_RE.search("WebhookSecret")
    assert SECRET_KEY_RE.search("bot_token")
    assert SECRET_KEY_RE.search("db_password")
    assert not SECRET_KEY_RE.search("allow_from")


# --- (e) security sweep over every discovered adapter ---------------------- #


def test_discovered_adapter_surface_is_locked() -> None:
    """The sweep below covers the frozen set of 16 built-in adapters."""
    assert len(ADAPTER_NAMES) == 16


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_split_never_leaks_secret_keys_in_values(name: str) -> None:
    """No SECRET_KEY_RE key may appear in ``values`` for any adapter."""
    section = _section_for(name)
    if section is None:
        # Missing SDK (e.g. matrix today): hints degrade to empty and masking
        # must still be fail-safe.
        assert channel_field_hints(name) == []
        section = {"webhook_secret": "dummy-secret-1234", "bot_token": "dummy-token"}

    values, secrets = split_values_secrets(name, section)
    assert [key for key in values if SECRET_KEY_RE.search(key)] == []
    assert set(values) | set(secrets) == set(section)
    assert not (set(values) & set(secrets))


# --- (f) unknown channel ---------------------------------------------------- #


def test_unknown_channel_has_no_hints() -> None:
    assert channel_field_hints("definitely_not_a_real_channel") == []
