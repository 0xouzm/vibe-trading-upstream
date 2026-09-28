"""Generate the broker capability matrix from the profile registry (#1626).

Which read tools, quote paths and order kinds each connector actually has
lived in per-connector code and tribal knowledge. The profiles already
declare it, so render them into the marked block in README.md and let a test
fail CI the moment the block and the registry disagree, so the table cannot
go stale.

Regenerate after touching connector profiles:

    python -m src.trading.capability_matrix
"""

from __future__ import annotations

import re
from pathlib import Path

from src.trading.profiles import BUILTIN_PROFILES

README_PATH = Path(__file__).resolve().parents[3] / "README.md"
_BEGIN = "<!-- BEGIN GENERATED broker-capability-matrix -->"
_END = "<!-- END GENERATED broker-capability-matrix -->"

_HEADER = """Which read and write paths each built-in connector actually exposes,
generated from the profile registry (`agent/src/trading/profiles.py`). An
order-capable profile still goes through the mandate gate: placement on live
funds is authorized only with a mandate in place.

| Connector | Environments | Modes | Read paths | Order paths |
| --- | --- | --- | --- | --- |
"""


def render() -> str:
    """One row per connector: envs, modes, and the union of read/order caps."""
    lines = [_HEADER]
    by_connector: dict[str, list] = {}
    for profile in BUILTIN_PROFILES:
        by_connector.setdefault(profile.connector, []).append(profile)
    for connector in sorted(by_connector):
        profiles = by_connector[connector]
        envs = "/".join(sorted({p.environment for p in profiles}))
        modes = "read-only" if all(p.readonly for p in profiles) else "read + order-capable"
        caps = {c for p in profiles for c in p.capabilities}
        reads = sorted(c for c in caps if c.endswith(".read"))
        orders = sorted(c for c in caps if c.startswith(("orders.", "copy.")))
        read_cell = ", ".join(f"`{c}`" for c in reads) or "none"
        order_cell = ", ".join(f"`{c}`" for c in orders) or "none"
        lines.append(f"| {connector} | {envs} | {modes} | {read_cell} | {order_cell} |")
    return "\n".join(lines) + "\n"


def render_block() -> str:
    return f"{_BEGIN}\n\n{render()}\n{_END}"


def sync_readme(readme: str) -> str:
    block = render_block()
    pattern = re.compile(re.escape(_BEGIN) + r".*?" + re.escape(_END), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError(f"README is missing the {_BEGIN} marker block")
    return pattern.sub(lambda _: block, readme)


def main() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    README_PATH.write_text(sync_readme(text), encoding="utf-8")
    print(f"synced {README_PATH} ({len(BUILTIN_PROFILES)} profiles)")


if __name__ == "__main__":
    main()
