from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionDenied(PermissionError):
    pass


class ApprovalRequired(PermissionError):
    pass


@dataclass(frozen=True)
class PermissionPolicy:
    read: Decision = Decision.ALLOW
    search: Decision = Decision.ALLOW
    write: Decision = Decision.ASK
    execute: Decision = Decision.ASK
    network: Decision = Decision.DENY

    def decision(self, action: str) -> Decision:
        try:
            return getattr(self, action)
        except AttributeError as exc:
            raise ValueError(f"unknown permission action: {action}") from exc

    def require(self, action: str, *, approved: bool = False) -> None:
        decision = self.decision(action)
        if decision is Decision.DENY:
            raise PermissionDenied(f"{action} permission denied")
        if decision is Decision.ASK and not approved:
            raise ApprovalRequired(f"{action} permission requires approval")
