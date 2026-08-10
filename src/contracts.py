"""Cloud package interface contracts."""

import re
from dataclasses import dataclass

SCRIPT_CAPABILITY_FIELDS = (
    "uses_llm",
    "uses_browser",
    "uses_files",
    "uses_file_context",
)

# ─── GCP resource label keys ─────────────────────────────────────────────────
# GCP label constraints: keys/values lowercase, max 63 chars, [a-z0-9_-] only.

SOURCE_HASH_LABEL = "lamia-source-hash"
LABEL_MANAGED = "lamia-managed"
LABEL_SCRIPT = "lamia-script"
LABEL_PROJECT_HASH = "lamia-project-hash"
LABEL_LAST_USED = "lamia-last-used"
LABEL_DEPLOY_MODE = "lamia-deploy-mode"
LABEL_REPO_URL = "lamia-repo-url"
LABEL_RESOURCE_TYPE = "lamia-resource-type"

STALE_RESOURCE_DAYS = 30

_LABEL_CLEAN_RE = re.compile(r"[^a-z0-9_-]")


def sanitize_label_value(value: str, max_len: int = 63) -> str:
    """Sanitize a string for use as a GCP label value.

    Lowercases, replaces disallowed chars (dots, slashes, spaces, etc.)
    with underscores, and truncates to max_len.
    """
    return _LABEL_CLEAN_RE.sub("_", value.lower())[:max_len]


@dataclass(frozen=True)
class FileSyncEntry:
    raw_path: str
    resolved_path: str
    bucket_key: str
