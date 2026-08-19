"""Anthropic model id parsing, catalog display naming, and version selection.

Anthropic has shipped two id schemes: pre-Claude-4 puts the version right
after "claude-" with the family after that (claude-3-5-haiku-20241022);
Claude 4 onward puts the family right after "claude-" (claude-haiku-4-5-...).

Dated snapshots are written with a trailing "-YYYYMMDD" almost everywhere (most
provider APIs, lamia config, this module's own ids above) except one place:
Vertex AI's wire format uses "@YYYYMMDD" instead (e.g.
claude-haiku-4-5@20251001). Parsing accepts both separators; to_vertex_id()
converts to the "@" form for the actual outgoing request.
"""
import re
from typing import Optional

# Claude 4 onward: version right after the family name. Minor is capped at
# 2 digits so it can't swallow an 8-digit date suffix as if it were a minor
# version (e.g. claude-sonnet-4-20250514 has no minor, just a date).
_MODEL_ID_RE_CURRENT = re.compile(r"^claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?:[@-]\d{8})?$")
# Pre-Claude-4 (e.g. claude-3-5-haiku-20241022, claude-3-opus-20240229):
# version comes right after "claude-", family after that.
_MODEL_ID_RE_LEGACY = re.compile(r"^claude-(\d+)(?:-(\d+))?-([a-z]+)(?:[@-]\d{8})?$")

_DATE_SUFFIX_RE = re.compile(r"-(\d{8})$")

_FAMILIES = ("opus", "sonnet", "haiku", "fable")


def model_garden_name(model_id: str) -> str:
    """Best-effort Model Garden search name for an Anthropic model id.

    Model Garden lists models under short marketing names (e.g. "Claude Sonnet
    4.5") rather than the dated API model id used in config and requests, so
    searching the raw id often finds nothing. Falls back to the raw id for
    anything that doesn't match a known id shape.

    Word order differs by era, matching Anthropic's own naming: pre-Claude-4
    names put the version before the family ("Claude 3.5 Sonnet"); Claude 4
    onward puts the family first ("Claude Sonnet 4.5"), matching the id shape.
    """
    match = _MODEL_ID_RE_CURRENT.match(model_id)
    if match:
        family, major, minor = match.groups()
        version = major if minor in (None, "0") else f"{major}.{minor}"
        return f"Claude {family.capitalize()} {version}"
    match = _MODEL_ID_RE_LEGACY.match(model_id)
    if match:
        major, minor, family = match.groups()
        version = major if minor is None else f"{major}.{minor}"
        return f"Claude {version} {family.capitalize()}"
    return model_id


def select_model(requested_model: str, available: list[str]) -> str:
    """Return `requested_model` if `available` has it exactly; otherwise the
    nearest available version in the same family (e.g. sonnet, opus), by
    version distance.

    Last resort, if that precise match fails or `requested_model` doesn't
    parse at all: guess the family by substring (catches catalog ids with
    an unexpected shape too, e.g. a preview suffix) and take the latest
    available in that family. Only if even that finds nothing does this
    return `requested_model` unchanged -- the resulting request then fails
    with an informative error rather than being silently dropped here.
    """
    if not available or requested_model in available:
        return requested_model

    requested = _family_version(requested_model)
    if requested is not None:
        family, (req_major, req_minor) = requested
        candidates = []
        for model_id in available:
            parsed = _family_version(model_id)
            if parsed is None or parsed[0] != family:
                continue
            _, (major, minor) = parsed
            distance = abs(major - req_major) * 1000 + abs(minor - req_minor)
            candidates.append((distance, -major, -minor, model_id))
        if candidates:
            candidates.sort()
            return candidates[0][3]
    else:
        family = _guess_family(requested_model)

    if family is None:
        return requested_model

    family_candidates = [model_id for model_id in available if family in model_id]
    if not family_candidates:
        return requested_model

    def _version_rank(model_id: str) -> tuple[int, int]:
        parsed = _family_version(model_id)
        return parsed[1] if parsed is not None else (0, 0)

    family_candidates.sort(key=_version_rank, reverse=True)
    return family_candidates[0]


def to_vertex_id(model_id: str) -> str:
    """Convert a dated snapshot id to Vertex AI's wire format ("@" before the
    date instead of "-"). Ids with no date, or already using "@", or that
    don't parse as a known Anthropic id shape at all, are returned unchanged.
    """
    if _parse_model_id(model_id) is None:
        return model_id
    return _DATE_SUFFIX_RE.sub(r"@\1", model_id)


def _parse_model_id(model_id: str) -> Optional[tuple[str, int, int]]:
    """Parse `model_id` into (family, major, minor) across both id schemes,
    or None if it matches neither. A single normalized shape lets
    family/version matching bridge the schemes -- e.g. finding that
    claude-haiku-4-5-... is the successor to the legacy claude-3-5-haiku-...
    """
    match = _MODEL_ID_RE_CURRENT.match(model_id)
    if match:
        family, major, minor = match.groups()
        return family, int(major), int(minor or 0)
    match = _MODEL_ID_RE_LEGACY.match(model_id)
    if match:
        major, minor, family = match.groups()
        return family, int(major), int(minor or 0)
    return None


def _family_version(model_id: str) -> Optional[tuple[str, tuple[int, int]]]:
    """Parse `model_id` into (family, (major, minor)), or None if unparseable."""
    parsed = _parse_model_id(model_id)
    if parsed is None:
        return None
    family, major, minor = parsed
    return family, (major, minor)


def _guess_family(model_id: str) -> Optional[str]:
    """Last-resort family guess via substring match, for ids that don't fit
    either _parse_model_id shape (e.g. an unrecognized future naming scheme)."""
    for family in _FAMILIES:
        if family in model_id:
            return family
    return None
