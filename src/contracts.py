"""Cloud package interface contracts."""

from dataclasses import dataclass

SCRIPT_CAPABILITY_FIELDS = (
    "uses_llm",
    "uses_browser",
    "uses_files",
    "uses_file_context",
)

SOURCE_HASH_LABEL = "lamia-source-hash"


@dataclass(frozen=True)
class FileSyncEntry:
    raw_path: str
    resolved_path: str
    bucket_key: str
