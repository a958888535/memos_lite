from __future__ import annotations

import re


_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    re.MULTILINE,
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)^\s*authorization\s*:\s*(?P<value>.+?)\s*$"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-=/+]{8,}")
_SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_SECRET_FIELD_RE = re.compile(
    r"""(?imx)
    (?P<prefix>
        (?P<key_quote>["']?)
        (?P<field>API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|PRIVATE_KEY|PASSWORD|SECRET|TOKEN)
        (?P=key_quote)
        \s*(?P<delimiter>[:=])\s*
    )
    (?P<value>
        (?P<value_quote>["']?)
        [^,\s}\]\r\n"']+
        (?P=value_quote)
    )
    """
)


def _redact_authorization_header(match: re.Match[str]) -> str:
    value = (match.group("value") or "").strip()
    if value.lower().startswith("bearer "):
        return "Authorization: [REDACTED_AUTH_HEADER] [REDACTED_BEARER_TOKEN]"
    return "Authorization: [REDACTED_AUTH_HEADER]"


def _redact_secret_field(match: re.Match[str]) -> str:
    field = str(match.group("field") or "")
    delimiter = str(match.group("delimiter") or "=")
    value = str(match.group("value") or "")

    if delimiter == "=":
        return f"{field}=[REDACTED_SECRET]"

    quote = ""
    if value[:1] in {"'", '"'}:
        quote = value[:1]
    return f"{match.group('prefix')}{quote}[REDACTED_SECRET]{quote}"


def redact_text(text: str) -> str:
    redacted = str(text or "")
    redacted = _PRIVATE_KEY_BLOCK_RE.sub("[REDACTED_PRIVATE_KEY_BLOCK]", redacted)
    redacted = _AUTH_HEADER_RE.sub(_redact_authorization_header, redacted)
    redacted = _BEARER_TOKEN_RE.sub("[REDACTED_BEARER_TOKEN]", redacted)
    redacted = _SK_KEY_RE.sub("[REDACTED_API_KEY]", redacted)
    redacted = _SECRET_FIELD_RE.sub(_redact_secret_field, redacted)
    return redacted


def contains_secret(text: str) -> bool:
    value = str(text or "")
    return any(
        pattern.search(value)
        for pattern in (
            _PRIVATE_KEY_BLOCK_RE,
            _AUTH_HEADER_RE,
            _BEARER_TOKEN_RE,
            _SK_KEY_RE,
            _SECRET_FIELD_RE,
        )
    )
