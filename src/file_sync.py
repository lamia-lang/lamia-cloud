"""Provider-agnostic file sync planning and safety scanning."""

import hashlib
from pathlib import Path

from lamia_cloud.contracts import FileSyncEntry

_SECRET_NAME_MARKERS = (
    ".env",
    "secret",
    "token",
    "credential",
    "private_key",
    "api_key",
    "auth",
)
_SECRET_CONTENT_MARKERS = (
    "api_key",
    "authorization:",
    "private key",
    "aws_secret_access_key",
    "-----begin",
)


def _make_bucket_key(
    raw_path: str,
    resolved_path: Path,
    project_root: Path,
) -> str:
    """Determine the GCS object key for a file.

    Relative paths stay relative (flat structure for teams).
    Absolute/~ paths use their full path as key.
    """
    if raw_path.startswith("~") or Path(raw_path).is_absolute():
        return str(resolved_path).replace("\\", "/").lstrip("/")
    try:
        rel = resolved_path.relative_to(project_root)
        key = str(rel).replace("\\", "/")
        return key[2:] if key.startswith("./") else key
    except ValueError:
        return str(resolved_path).replace("\\", "/").lstrip("/")


def _resolve_local_path(raw_path: str, script_dir: Path, local_home: Path) -> Path:
    if raw_path.startswith("~"):
        expanded = raw_path.replace("~", str(local_home), 1)
        return Path(expanded).expanduser().resolve()
    if Path(raw_path).is_absolute():
        return Path(raw_path).resolve()
    return (script_dir / raw_path).resolve()


def _warn_if_secret(path: Path) -> bool:
    lower_name = str(path).lower()
    if any(marker in lower_name for marker in _SECRET_NAME_MARKERS):
        return True
    if not path.exists() or not path.is_file():
        return False
    try:
        sample = path.read_text(errors="ignore").lower()
    except Exception:
        return False
    return any(marker in sample for marker in _SECRET_CONTENT_MARKERS)


def _iter_directory_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            out.append(p)
    return out


def build_file_sync_plan(
    files_context_paths: list[str],
    project_root: Path,
    local_home: Path | None = None,
) -> list[FileSyncEntry]:
    """Resolve with files(...) paths to entries ready for cloud upload."""
    local_home = local_home or Path.home()
    base_dir = project_root.resolve()
    seen: dict[tuple[str, str], FileSyncEntry] = {}

    for raw_path in files_context_paths:
        resolved = _resolve_local_path(raw_path, base_dir, local_home)
        if not resolved.exists():
            raise FileNotFoundError(f"with files() path not found: {raw_path} (resolved: {resolved})")

        if resolved.is_dir():
            root_key = _make_bucket_key(raw_path, resolved, base_dir)
            for child in _iter_directory_files(resolved):
                if _warn_if_secret(child):
                    raise ValueError(f"Potential secret file in cloud sync: {child}")
                key = f"{root_key.rstrip('/')}/{child.relative_to(resolved).as_posix()}"
                entry = FileSyncEntry(raw_path=raw_path, resolved_path=str(child), bucket_key=key)
                seen[(str(child), key)] = entry
        else:
            if _warn_if_secret(resolved):
                raise ValueError(f"Potential secret file in cloud sync: {resolved}")
            key = _make_bucket_key(raw_path, resolved, base_dir)
            seen[(str(resolved), key)] = FileSyncEntry(raw_path=raw_path, resolved_path=str(resolved), bucket_key=key)

    return list(seen.values())


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
