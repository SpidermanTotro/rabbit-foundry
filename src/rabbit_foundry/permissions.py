from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Permission(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionPolicy:
    read: Permission = Permission.ASK
    search: Permission = Permission.ASK
    edit: Permission = Permission.ASK
    shell: Permission = Permission.ASK
    network: Permission = Permission.DENY
    external_directory: Permission = Permission.DENY

    @classmethod
    def dry_run(cls) -> "PermissionPolicy":
        return cls(
            read=Permission.ALLOW,
            search=Permission.ALLOW,
            edit=Permission.DENY,
            shell=Permission.DENY,
            network=Permission.DENY,
            external_directory=Permission.DENY,
        )

    @classmethod
    def supervised(cls) -> "PermissionPolicy":
        return cls(
            read=Permission.ALLOW,
            search=Permission.ALLOW,
            edit=Permission.ASK,
            shell=Permission.ASK,
            network=Permission.ASK,
            external_directory=Permission.DENY,
        )
