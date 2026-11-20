from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SanitizedContent:
    content: str
    injection_flagged: bool
    matched_patterns: tuple[str, ...]


class UntrustedContentPolicy:
    _patterns = (
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
        re.compile(r"change\s+(the\s+)?model\s+configuration", re.IGNORECASE),
        re.compile(r"system\s+prompt", re.IGNORECASE),
        re.compile(r"allowed_tools", re.IGNORECASE),
        re.compile(r"调用工具"),
        re.compile(r"读取密钥"),
        re.compile(r"修改(?:系统|模型|工具)?配置"),
        re.compile(r"执行(?:命令|shell)"),
    )

    def detect(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._patterns)

    def sanitize(self, text: str) -> SanitizedContent:
        sanitized = text
        matches: list[str] = []
        for pattern in self._patterns:
            if pattern.search(sanitized):
                matches.append(pattern.pattern)
                sanitized = pattern.sub("[UNTRUSTED_INSTRUCTION_REDACTED]", sanitized)
        return SanitizedContent(
            content=sanitized,
            injection_flagged=bool(matches),
            matched_patterns=tuple(matches),
        )
