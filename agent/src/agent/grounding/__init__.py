"""Run-scoped identity and numeric evidence gates for the main agent loop.

The language model remains responsible for research and explanation, but three
facts are structural rather than advisory:

* a market-data consumer may only use an identity that was locked before the
  current assistant tool-call batch started;
* a final price claim may not contradict the full, untruncated tool result; and
* a figure may not be attached to an instrument that no tool call in this run
  ever passed in or returned.

Those are the mechanically decidable parts of the agent's output principles.
The rest of that contract — "state the as-of", "analysis, not advice", "refuse
out loud" — stays in the system prompt on purpose: see
``policies._validate_price_claims`` and the package tests for why a regex gate
on them rejects correct answers.

The package deliberately contains no provider or tool-registry dependencies so
its state machine and final-answer checks remain deterministic and testable.
The split follows the data flow: :mod:`identity` locks what a number is about,
:mod:`evidence` records what the run actually observed, :mod:`figures` finds
the numbers in a draft, :mod:`policies` decides whether each one is grounded,
:mod:`release` turns a rejection into a correction or a discounted release, and
:mod:`ledger` is the facade the agent loop drives.
"""

from __future__ import annotations

from src.agent.resolution_context import IdentityConstraint, ResolutionContext

from src.agent.grounding.identity import IdentityRecord, ToolAuthorization
from src.agent.grounding.evidence import EvidenceRecord
from src.agent.grounding.policies import ValidationResult
from src.agent.grounding.ledger import GROUNDING_ARTIFACT, GroundingLedger

__all__ = [
    "GROUNDING_ARTIFACT",
    "EvidenceRecord",
    "GroundingLedger",
    "IdentityConstraint",
    "IdentityRecord",
    "ResolutionContext",
    "ToolAuthorization",
    "ValidationResult",
]
