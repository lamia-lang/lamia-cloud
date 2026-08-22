"""GCP Vertex AI service implementation.

Routes LLM calls through Vertex AI using ADC. On Cloud Run, authentication
is automatic via the metadata server. Locally, uses `gcloud auth application-default login`.

Three request formats, chosen automatically:
- Google  → generateContent (Gemini-specific)
- Anthropic → rawPredict with anthropic_version header (Anthropic-specific)
- Any other provider → rawPredict with OpenAI-compatible chat completions

No static allowlist: any provider that appears in config.yaml and isn't
"google" or "anthropic" is sent via the generic rawPredict path.  A small
alias table normalises common shorthand names to Vertex publisher ids
(e.g. "mistral" → "mistralai").
"""
import asyncio
import difflib
import json
import logging
import os
import re
import urllib.request
from typing import Optional, Dict, Any

import aiohttp
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

from lamia_cloud.gcp.llm import anthropic_mapper
from lamia_cloud.gcp.storage_utils import ensure_bucket
from lamia_cloud.interfaces import CloudLLM
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse

logger = logging.getLogger(__name__)

ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
DEFAULT_REGION = "us-central1"
# Recommended by Google for Claude on Vertex: best availability, no pricing
# premium (unlike a pinned region or multi-region), and no allowlist of
# regions to maintain -- any region/multi-region the user configures
# explicitly is honored as-is in _region_for_provider.
ANTHROPIC_DEFAULT_REGION = "global"

# Regions confirmed to not serve Anthropic models on Vertex AI. Live-tested:
# Vertex rejects rawPredict there with "is not servable in region <region>".
# Independent of DEFAULT_REGION, which is just this codebase's config
# fallback and can change for unrelated reasons.
ANTHROPIC_UNAVAILABLE_REGIONS = frozenset({"us-central1"})
FALLBACK_GEMINI_MODELS = {
    # Used only when model listing is unavailable.
    "strong": "gemini-2.5-pro",
    "medium": "gemini-2.5-flash",
    "light": "gemini-2.5-flash-lite",
}
OPENAI_MODEL_FAMILIES = {
    "strong": (
        "o4",
        "o3",
        "o2",
        "o1",
        "gpt-5",
        "gpt-4.5",
        "gpt-4.1",
    ),
    "medium": (
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
    ),
    "light": (
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-3.5",
    ),
}

# Providers whose models can only be reached via Gemini tier mapping
# because they have no Vertex AI publisher.  Every other provider —
# including any future Model Garden partner — goes through rawPredict.
_NON_VERTEX_PROVIDERS = frozenset({"openai"})

# Normalise common config-level shorthand names to the actual Vertex AI
# publisher id.  Any provider NOT listed here is used as its own publisher
# id verbatim — no static allowlist needed.
PUBLISHER_ALIASES: dict[str, str] = {
    "mistral": "mistralai",
}

# Backward-compatible exports used by existing tests/integrations.
VERTEX_API_VERSION = ANTHROPIC_VERTEX_VERSION
VERTEX_REGION = DEFAULT_REGION


def is_on_gcp() -> bool:
    """Detect if running inside GCP via the metadata server."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        resp = urllib.request.urlopen(req, timeout=1)
        return resp.status == 200
    except Exception:
        return False


_MODEL_MAP_KEY = "lamia-vertex-model-map"
_REGIONS_MAP_KEY = "lamia-vertex-model-regions"

_FALLBACK_PROBE_REGIONS = [
    "us-east5",
    "us-central1",
    "global",
    "us",
    "eu",
    "europe-west1",
    "asia-southeast1",
]


def _model_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _split_model_key(key: str) -> Optional[tuple[str, str]]:
    provider, sep, model = key.partition(":")
    return (provider, model) if sep and provider and model else None


def get_verified_vertex_models(project_id: str) -> set[tuple[str, str]]:
    """(provider, model) pairs already confirmed accessible on Vertex AI for this project.

    Stored as JSON in a GCS bucket, so the cache is shared across
    every team member and CI run against that project, not just one checkout.
    """
    pairs = set()
    for key in _read_verified_models(project_id):
        pair = _split_model_key(key)
        if pair:
            pairs.add(pair)
    return pairs


def remember_verified_vertex_models(project_id: str, models: set[tuple[str, str]]) -> None:
    """Record confirmed Vertex AI model access."""
    if not models:
        return
    updates = {_model_key(p, m): _model_key(p, m) for p, m in models}
    _update_verified_models(project_id, updates)


def _read_verified_models(project_id: str) -> dict[str, str]:
    bucket_name = _lamia_state_bucket_name(project_id)
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_VERIFIED_MODELS_BLOB)
        data = json.loads(blob.download_as_bytes())
        return data.get(_MODEL_MAP_KEY, {})
    except Exception as exc:
        logger.warning(
            f"Could not read model cache from gs://{bucket_name}/{_VERIFIED_MODELS_BLOB}: {exc}"
        )
        return {}


def _get_cached_resolution(project_id: str, provider: str, model: str) -> Optional[tuple[str, str]]:
    """Look up the (provider, model) `(provider, model)` was previously
    confirmed to resolve to, if any -- itself when verified as requested,
    or a different model (same provider or a different one entirely, e.g.
    a partner with no Vertex publisher mapped to a Gemini tier) when
    resolution substituted one."""
    resolved = _read_verified_models(project_id).get(_model_key(provider, model))
    return _split_model_key(resolved) if resolved else None


def _remember_resolved_models(project_id: str, mapping: dict[tuple[str, str], tuple[str, str]]) -> None:
    """Persist the full requested -> resolved correlation for confirmed pairs.

    Internal to this module -- CloudDeployer.remember_verified_model_access
    only ever sees a flat set (see remember_verified_vertex_models), since
    nothing outside this file needs to know what a pair resolved to.
    """
    if not mapping:
        return
    updates = {
        _model_key(*requested): _model_key(*resolved)
        for requested, resolved in mapping.items()
    }
    _update_verified_models(project_id, updates)


def _update_cache_section(project_id: str, section_key: str, updates: dict[str, str],
                          *, identity_safe: bool = True) -> None:
    """Merge `updates` into one section of the verified-models cache blob.

    When *identity_safe* is True (the default, used for model maps), a real
    substitution (value != key) always overwrites; an identity entry
    (value == key) only fills in a key that isn't already known.

    When *identity_safe* is False (used for region maps), every update is
    written unconditionally.
    """
    try:
        bucket = ensure_bucket(project_id, _lamia_state_bucket_name(project_id))
        blob = bucket.blob(_VERIFIED_MODELS_BLOB)
        for _attempt in range(2):
            if blob.exists():
                blob.reload()
                generation = blob.generation
                full_data = json.loads(blob.download_as_bytes())
            else:
                generation = 0
                full_data = {}
            section = full_data.get(section_key, {})
            for key, value in updates.items():
                if not identity_safe or value != key or key not in section:
                    section[key] = value
            full_data[section_key] = section
            payload = json.dumps(full_data).encode()
            try:
                blob.upload_from_string(
                    payload, content_type="application/json",
                    if_generation_match=generation,
                )
                return
            except PreconditionFailed:
                continue
    except Exception as exc:
        logger.warning(f"Failed to update cache section {section_key} on project {project_id}: {exc}")


def _update_verified_models(project_id: str, updates: dict[str, str]) -> None:
    _update_cache_section(project_id, _MODEL_MAP_KEY, updates, identity_safe=True)


def _get_cached_model_regions(project_id: str) -> dict[str, str]:
    """Read the model→region cache: ``{"meta:llama-4-...": "us-east5", ...}``."""
    bucket_name = _lamia_state_bucket_name(project_id)
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_VERIFIED_MODELS_BLOB)
        data = json.loads(blob.download_as_bytes())
        return data.get(_REGIONS_MAP_KEY, {})
    except Exception as exc:
        logger.warning(
            f"Could not read region cache from gs://{bucket_name}/{_VERIFIED_MODELS_BLOB}: {exc}"
        )
        return {}


def _remember_model_regions(project_id: str, regions: dict[str, str]) -> None:
    """Persist discovered model→region mappings."""
    if not regions:
        return
    _update_cache_section(project_id, _REGIONS_MAP_KEY, regions, identity_safe=False)


def _resolve_publisher(provider: str) -> str:
    """Map a config-level provider name to its Vertex AI publisher id.

    Applies alias normalisation (e.g. "mistral" → "mistralai"), then
    returns the provider as-is — every Vertex partner IS its own publisher.
    """
    return PUBLISHER_ALIASES.get(provider, provider)


def _extract_version_score(model_id: str) -> tuple[int, int]:
    """Return (major, minor) tuple for sorting Gemini model recency."""
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", model_id)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


def _model_similarity(requested: str, candidate: str) -> float:
    """Score how similar two model names are (0..1, higher = better)."""
    a, b = requested.lower(), candidate.lower()
    if a == b:
        return 1.0
    toks_a = set(re.split(r"[-_.]", a))
    toks_b = set(re.split(r"[-_.]", b))
    if not toks_a:
        return 0.0
    token_overlap = len(toks_a & toks_b) / max(len(toks_a), len(toks_b))
    seq_ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return 0.6 * token_overlap + 0.4 * seq_ratio


def _classify_openai_tier(model: str) -> str:
    """Classify OpenAI models by family prefix (configurable above-the-fold)."""
    model = model.lower()

    best_tier = "light"
    best_prefix_len = -1
    for tier, prefixes in OPENAI_MODEL_FAMILIES.items():
        for prefix in prefixes:
            if model.startswith(prefix) and len(prefix) > best_prefix_len:
                best_tier = tier
                best_prefix_len = len(prefix)

    if best_prefix_len >= 0:
        return best_tier
    return "light"


def _get_project_id() -> str:
    """Resolve GCP project ID from environment or metadata server."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if project_id:
        return project_id
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        resp = urllib.request.urlopen(req, timeout=2)
        return resp.read().decode().strip()
    except Exception:
        return ""


def _get_access_token() -> str:
    """Get an OAuth2 access token via google-auth ADC."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def _region_for_provider(provider: str, configured_region: str) -> str:
    """Resolve the region for a provider's Vertex AI calls.

    Anthropic gets routed to ANTHROPIC_DEFAULT_REGION whenever the configured
    region is one of ANTHROPIC_UNAVAILABLE_REGIONS. Any other region,
    multi-region ("us"/"eu"), or "global" the user configures explicitly is
    used as-is -- no allowlist to keep in sync with Google's rollout.
    """
    if provider == "anthropic" and configured_region in ANTHROPIC_UNAVAILABLE_REGIONS:
        return ANTHROPIC_DEFAULT_REGION
    return configured_region


def _vertex_endpoint_host(region: str) -> str:
    """Hostname for a Vertex AI region: global, multi-region ("us"/"eu"), or
    a specific region (e.g. "us-east5") each use a different host pattern."""
    if region == "global":
        return "aiplatform.googleapis.com"
    if region in ("us", "eu"):
        return f"aiplatform.{region}.rep.googleapis.com"
    return f"{region}-aiplatform.googleapis.com"


_LAMIA_STATE_BUCKET_SUFFIX = "-lamia-state"
_VERIFIED_MODELS_BLOB = "lamia-cache/verified-models.json"


def _lamia_state_bucket_name(project_id: str) -> str:
    return f"{project_id}{_LAMIA_STATE_BUCKET_SUFFIX}"


class VertexLLM(CloudLLM):
    """Cloud LLM backed by GCP Vertex AI.

    Automatically routes to the correct endpoint based on provider:
    - anthropic → rawPredict (with anthropic_version)
    - google → generateContent (Gemini format)
    """

    def __init__(self, region: str = DEFAULT_REGION, project_id: str | None = None) -> None:
        self.project_id = project_id or _get_project_id()
        self.configured_region = region
        # Keep old public attribute name for compatibility.
        self.region = region
        self._session: Optional[aiohttp.ClientSession] = None
        self._publisher_models_cache: dict[str, list[str]] = {}
        self._anthropic_version = ANTHROPIC_VERTEX_VERSION
        self._model_regions: dict[str, str] = {}

    @classmethod
    def from_config(cls, cloud_cfg: dict) -> "VertexLLM":
        project_id = cloud_cfg.get("project_id")
        if not project_id:
            raise ValueError("cloud.project_id is required in config.yaml.")
        region = cloud_cfg.get("location", DEFAULT_REGION)
        return cls(region=region, project_id=project_id)

    def is_available(self) -> bool:
        return is_on_gcp()

    async def generate(self, request: CloudLLMRequest) -> CloudLLMResponse:
        if not self._model_regions:
            self._model_regions = _get_cached_model_regions(self.project_id)

        cached = _get_cached_resolution(
            self.project_id, request.provider, request.model,
        )
        if cached:
            resolved_provider, resolved_model = cached
            if resolved_provider != request.provider or resolved_model != request.model:
                logger.info(
                    f"Using cached mapping: {request.provider}/{request.model} -> "
                    f"{resolved_provider}/{resolved_model}"
                )
                request = CloudLLMRequest(
                    prompt=request.prompt,
                    model=resolved_model,
                    provider=resolved_provider,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    response_schema=request.response_schema,
                )

        original_provider = request.provider
        original_model = request.model
        request = self._resolve_model(request)

        if request.provider == "google" and request.model.startswith("__auto_tier__:"):
            tier = request.model.split(":", 1)[1]
            mapped_model = await self._select_google_model_for_tier(tier)
            logger.info(
                f"Cloud model mapping: {original_provider}/{original_model} -> "
                f"google/{mapped_model} (tier: {tier})"
            )
            request = CloudLLMRequest(
                prompt=request.prompt,
                model=mapped_model,
                provider="google",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                response_schema=request.response_schema,
            )

        url, payload = self._build_request(request)

        token = _get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        if self._session is None:
            self._session = aiohttp.ClientSession()

        async with self._session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()

                if request.provider == "anthropic":
                    hinted = self._extract_anthropic_version_hint(error_text)
                    if hinted and hinted != self._anthropic_version:
                        self._anthropic_version = hinted
                        retry_url, retry_payload = self._build_anthropic_request(
                            request, self._anthropic_version
                        )
                        async with self._session.post(
                            retry_url, json=retry_payload, headers=headers
                        ) as retry_response:
                            if retry_response.status == 200:
                                data = await retry_response.json()
                                return self._parse_anthropic_response(data, request.model)
                            error_text = await retry_response.text()
                            self._handle_api_error(retry_response.status, error_text, request)
                            raise RuntimeError(
                                f"Vertex AI error ({retry_response.status}): {error_text}"
                            )

                if response.status == 404:
                    model_key = _model_key(request.provider, request.model)
                    current_region = self._model_regions.get(
                        model_key,
                        _region_for_provider(request.provider, self.configured_region),
                    )
                    if current_region not in ("global", "us", "eu"):
                        for fallback in ("global", "us"):
                            self._model_regions[model_key] = fallback
                            fb_url, fb_payload = self._build_request(request)
                            async with self._session.post(
                                fb_url, json=fb_payload, headers=headers
                            ) as fb_resp:
                                if fb_resp.status == 200:
                                    logger.info(
                                        f"Model {request.provider}/{request.model} "
                                        f"not in {current_region}, found in {fallback}"
                                    )
                                    data = await fb_resp.json()
                                    return self._parse_response(data, request)
                                if fb_resp.status != 404:
                                    break
                        self._model_regions.pop(model_key, None)

                self._handle_api_error(response.status, error_text, request)
                raise RuntimeError(f"Vertex AI error ({response.status}): {error_text}")

            data = await response.json()
            return self._parse_response(data, request)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def check_model_access(
        self, models: list[tuple[str, str]]
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        dict[tuple[str, str], list[str]],
        dict[tuple[str, str], str],
    ]:
        """Check all models (every provider) for accessibility.

        Attempts auto-enable for partner services when possible.
        Returns (missing, verified, suggestions, needs_terms).

        ``needs_terms`` maps models that exist in the catalog but require
        manual EULA / terms acceptance to their Model Garden page URL.
        """
        if not models:
            return [], [], {}, {}
        return asyncio.run(self._check_all_models(models))

    def catalog_display_name(self, provider: str, model: str) -> str:
        if provider != "anthropic":
            return model
        return anthropic_mapper.model_garden_name(model)

    def model_catalog_url(self) -> str:
        return f"https://console.cloud.google.com/agent-platform/model-garden?project={self.project_id}"

    def model_page_url(self, provider: str, model: str) -> str:
        publisher = _resolve_publisher(provider)
        return (
            f"https://console.cloud.google.com/vertex-ai/publishers/"
            f"{publisher}/model-garden/{model}?project={self.project_id}"
        )

    def _resolve_model(self, request: CloudLLMRequest) -> CloudLLMRequest:
        """Route provider to the correct Vertex AI path.

        Only providers that have NO Vertex publisher (openai, openrouter)
        are mapped to a Gemini tier.  Everything else — Anthropic, Google,
        Mistral, Meta, Moonshot, Qwen, whatever — passes through as-is.
        """
        if request.provider in _NON_VERTEX_PROVIDERS:
            tier = _classify_openai_tier(request.model)
            return CloudLLMRequest(
                prompt=request.prompt,
                model=f"__auto_tier__:{tier}",
                provider="google",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                response_schema=request.response_schema,
            )
        return request

    async def _load_publisher_models(self, publisher: str) -> list[str]:
        """List available `publisher` models from Vertex AI's live catalog."""
        if publisher in self._publisher_models_cache:
            return self._publisher_models_cache[publisher]

        token = _get_access_token()

        # Two endpoint styles: v1 project-scoped works for Google models,
        # v1beta1 global works for partner publishers (mistralai, meta, etc.).
        region = _region_for_provider(publisher, self.configured_region)
        urls = [
            (
                f"https://{_vertex_endpoint_host(region)}/v1/"
                f"projects/{self.project_id}/locations/{region}/publishers/{publisher}/models",
                {"Authorization": f"Bearer {token}"},
            ),
            (
                f"https://us-central1-aiplatform.googleapis.com/v1beta1/"
                f"publishers/{publisher}/models",
                {"Authorization": f"Bearer {token}", "x-goog-user-project": self.project_id},
            ),
        ]

        models: list[str] = []
        try:
            async with aiohttp.ClientSession() as discovery_session:
                for url, headers in urls:
                    async with discovery_session.get(url, headers=headers) as response:
                        if response.status != 200:
                            continue
                        data = await response.json()
                        raw_models = data.get("publisherModels") or data.get("models") or []
                        for item in raw_models:
                            if isinstance(item, str):
                                model_id = item.rsplit("/models/", 1)[-1]
                                models.append(model_id)
                                continue
                            if not isinstance(item, dict):
                                continue
                            name = item.get("name", "")
                            model_id = item.get("model", "") or item.get("modelId", "")
                            if not model_id and "/models/" in name:
                                model_id = name.rsplit("/models/", 1)[-1]
                            if model_id:
                                models.append(model_id)
                    if models:
                        break
        except Exception:
            pass

        result = sorted(set(models))
        self._publisher_models_cache[publisher] = result
        return result

    def _google_tier_candidates(self, tier: str, available: list[str]) -> list[str]:
        """Return Gemini model candidates for a tier, sorted best-first.

        Filters out non-text models (tts, embedding, image, etc.) and sorts
        by version score descending so the newest model comes first.
        """
        _SKIP_SUFFIXES = ("tts", "embedding", "image", "audio", "robotics", "computer-use")

        def _is_text_model(m: str) -> bool:
            return m.startswith("gemini-") and not any(s in m for s in _SKIP_SUFFIXES)

        if tier == "strong":
            candidates = [m for m in available if _is_text_model(m) and "pro" in m and "preview" not in m]
            if not candidates:
                candidates = [m for m in available if _is_text_model(m) and "pro" in m]
        elif tier == "light":
            candidates = [m for m in available if _is_text_model(m) and ("flash-lite" in m or "lite" in m) and "preview" not in m]
        else:
            candidates = [m for m in available if _is_text_model(m) and "flash" in m and "lite" not in m and "preview" not in m]

        candidates.sort(key=_extract_version_score, reverse=True)
        return candidates

    async def _select_google_model_for_tier(self, tier: str) -> str:
        """Select best available Gemini model for the requested tier."""
        candidates = await self._ranked_google_candidates(tier)
        return candidates[0] if candidates else FALLBACK_GEMINI_MODELS[tier]

    async def _ranked_google_candidates(self, tier: str) -> list[str]:
        """Return all Gemini candidates for a tier, best-first, with fallback appended."""
        available = await self._load_publisher_models("google")
        candidates = self._google_tier_candidates(tier, available) if available else []
        fallback = FALLBACK_GEMINI_MODELS[tier]
        if fallback not in candidates:
            candidates.append(fallback)
        return candidates

    async def _select_anthropic_model(self, requested_model: str) -> str:
        """Resolve `requested_model` against Vertex's live Anthropic catalog.

        See anthropic_mapper.select_model for the exact-match / nearest-version
        / family-substring fallback chain.
        """
        available = await self._load_publisher_models("anthropic")
        return anthropic_mapper.select_model(requested_model, available)

    def _build_request(self, request: CloudLLMRequest) -> tuple[str, Dict[str, Any]]:
        """Dispatch to the correct request builder based on provider."""
        if request.provider == "anthropic":
            return self._build_anthropic_request(request, self._anthropic_version)
        if request.provider == "google":
            return self._build_google_request(request)
        return self._build_partner_request(request)

    def _parse_response(self, data: dict, request: CloudLLMRequest) -> CloudLLMResponse:
        """Dispatch to the correct response parser based on provider."""
        if request.provider == "anthropic":
            return self._parse_anthropic_response(data, request.model)
        if request.provider == "google":
            return self._parse_google_response(data, request.model)
        return self._parse_partner_response(data, request.model)

    def _build_anthropic_request(
        self,
        request: CloudLLMRequest,
        anthropic_version: str,
    ) -> tuple[str, Dict[str, Any]]:
        """Build rawPredict request for Anthropic models on Vertex."""
        region = _region_for_provider("anthropic", self.configured_region)
        vertex_model_id = anthropic_mapper.to_vertex_id(request.model)
        url = (
            f"https://{_vertex_endpoint_host(region)}/v1/"
            f"projects/{self.project_id}/locations/{region}/"
            f"publishers/anthropic/models/{vertex_model_id}:rawPredict"
        )

        payload: Dict[str, Any] = {
            "anthropic_version": anthropic_version,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
        }

        if request.top_p is not None and request.temperature is None:
            payload["top_p"] = request.top_p
        elif request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.response_schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": request.response_schema}
            }

        return url, payload

    def _build_partner_request(
        self, request: CloudLLMRequest,
    ) -> tuple[str, Dict[str, Any]]:
        """Build rawPredict request for any Vertex partner (OpenAI-compatible)."""
        publisher = _resolve_publisher(request.provider)
        model_key = _model_key(request.provider, request.model)
        region = self._model_regions.get(
            model_key,
            _region_for_provider(request.provider, self.configured_region),
        )
        url = (
            f"https://{_vertex_endpoint_host(region)}/v1/"
            f"projects/{self.project_id}/locations/{region}/"
            f"publishers/{publisher}/models/{request.model}:rawPredict"
        )

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        return url, payload

    def _build_google_request(self, request: CloudLLMRequest) -> tuple[str, Dict[str, Any]]:
        """Build generateContent request for Google models (Gemini) on Vertex."""
        model_key = _model_key("google", request.model)
        region = self._model_regions.get(
            model_key,
            _region_for_provider("google", self.configured_region),
        )
        url = (
            f"https://{_vertex_endpoint_host(region)}/v1/"
            f"projects/{self.project_id}/locations/{region}/"
            f"publishers/google/models/{request.model}:generateContent"
        )

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {"maxOutputTokens": request.max_tokens},
        }

        if request.temperature is not None:
            payload["generationConfig"]["temperature"] = request.temperature
        if request.top_p is not None:
            payload["generationConfig"]["topP"] = request.top_p

        if request.response_schema is not None:
            payload["generationConfig"]["responseSchema"] = request.response_schema
            payload["generationConfig"]["responseMimeType"] = "application/json"

        return url, payload

    def _parse_anthropic_response(self, data: dict, model: str) -> CloudLLMResponse:
        return CloudLLMResponse(
            text=data["content"][0]["text"],
            model=model,
            usage={
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0)
                ),
            },
            raw=data,
        )

    def _parse_partner_response(self, data: dict, model: str) -> CloudLLMResponse:
        """Parse OpenAI-compatible chat completions response from partner models."""
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage_raw = data.get("usage", {})
        return CloudLLMResponse(
            text=text,
            model=model,
            usage={
                "input_tokens": usage_raw.get("prompt_tokens", 0),
                "output_tokens": usage_raw.get("completion_tokens", 0),
                "total_tokens": usage_raw.get("total_tokens", 0),
            },
            raw=data,
        )

    def _parse_google_response(self, data: dict, model: str) -> CloudLLMResponse:
        # Preferred Vertex Gemini shape.
        if "candidates" in data:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [{}])
            text = parts[0].get("text", "") if parts else ""
            usage_meta = data.get("usageMetadata", {})
            usage = {
                "input_tokens": usage_meta.get("promptTokenCount", 0),
                "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            }
        else:
            # Backward/test compatibility fallback (Anthropic-like shape).
            text = data.get("content", [{}])[0].get("text", "")
            usage_raw = data.get("usage", {})
            usage = {
                "input_tokens": usage_raw.get("input_tokens", usage_raw.get("prompt_tokens", 0)),
                "output_tokens": usage_raw.get("output_tokens", usage_raw.get("completion_tokens", 0)),
                "total_tokens": (
                    usage_raw.get("total_tokens")
                    or usage_raw.get("input_tokens", usage_raw.get("prompt_tokens", 0))
                    + usage_raw.get("output_tokens", usage_raw.get("completion_tokens", 0))
                ),
            }
        return CloudLLMResponse(
            text=text,
            model=model,
            usage=usage,
            raw=data,
        )

    @staticmethod
    def _extract_anthropic_version_hint(error_text: str) -> str | None:
        """Extract a newer anthropic_version from API error text, if present."""
        versions = re.findall(r"vertex-\d{4}-\d{2}-\d{2}", error_text)
        return versions[-1] if versions else None

    def _handle_api_error(self, status: int, error_text: str, request: CloudLLMRequest) -> None:
        """Log actionable guidance for common Vertex AI errors."""
        if status == 404:
            model_key = _model_key(request.provider, request.model)
            region = self._model_regions.get(
                model_key,
                _region_for_provider(request.provider, self.configured_region),
            )
            model_garden_url = self.model_catalog_url()
            search_hint = self.catalog_display_name(request.provider, request.model)
            logger.error(
                f"Model {request.provider}/{request.model} not found in "
                f"region {region}. Search \"{search_hint}\" in Model Garden:\n"
                f"  {model_garden_url}"
            )
            try:
                import webbrowser
                webbrowser.open(model_garden_url)
            except Exception:
                pass
        elif status == 429:
            logger.warning(
                f"Quota exceeded for {request.provider}/{request.model}. "
                f"Request a quota increase: "
                f"https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai"
            )
        elif status == 403 and "SERVICE_DISABLED" in error_text:
            logger.error(
                f"Vertex AI API not enabled. Enabling automatically..."
            )

    async def _check_all_models(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        dict[tuple[str, str], list[str]],
        dict[tuple[str, str], str],
    ]:
        """Pre-deploy model access check for ALL providers.

        1. Resolve Anthropic model names against live catalog; resolve
           providers with no Vertex publisher (openai) to a Gemini tier,
           the same way generate() does for real calls.
        2. Probe every model in parallel.
        3. Auto-enable partner services for inaccessible models.
        4. Re-probe after enable.
        5. For still-missing models, automatically try the closest
           available names -- the first one that's accessible becomes the
           resolved substitute.

        Returns (missing, verified, suggestions, needs_terms).

        ``needs_terms`` maps (provider, model) to the Model Garden page
        URL for models that exist in the catalog but require manual EULA
        or terms acceptance before they can be used.
        """
        try:
            resolved_pairs = []
            for provider, model in pairs:
                if provider == "anthropic":
                    resolved_model = await self._select_anthropic_model(model)
                    if resolved_model != model:
                        logger.warning(
                            f"Cloud model mapping: anthropic/{model} -> "
                            f"anthropic/{resolved_model} (exact version unavailable)"
                        )
                    resolved_pairs.append(("anthropic", resolved_model))
                elif provider in _NON_VERTEX_PROVIDERS:
                    tier = _classify_openai_tier(model)
                    mapped_model = await self._select_google_model_for_tier(tier)
                    resolved_pairs.append(("google", mapped_model))
                else:
                    resolved_pairs.append((provider, model))

            results = list(await asyncio.gather(
                *(self._probe_model_access(p, m) for p, m in resolved_pairs)
            ))

            # Auto-enable partner services for inaccessible models
            for idx, ((provider, model), accessible) in enumerate(
                zip(resolved_pairs, results)
            ):
                if accessible is not False:
                    continue
                enabled = await self._auto_enable_partner_model(provider, model)
                if enabled:
                    re_probe = await self._probe_model_access(provider, model)
                    if re_probe is True:
                        results[idx] = True

            # _NON_VERTEX_PROVIDERS whose first Gemini candidate failed:
            # try remaining candidates before giving up.
            for idx, ((orig_provider, orig_model), (res_p, res_m), accessible) in enumerate(
                zip(pairs, resolved_pairs, results)
            ):
                if orig_provider not in _NON_VERTEX_PROVIDERS or accessible is True:
                    continue
                tier = _classify_openai_tier(orig_model)
                for candidate in await self._ranked_google_candidates(tier):
                    if candidate == res_m:
                        continue
                    probe = await self._probe_model_access("google", candidate)
                    if probe is True:
                        results[idx] = True
                        resolved_pairs[idx] = ("google", candidate)
                        break

            # Log final mapping for _NON_VERTEX_PROVIDERS (once, after retries)
            for (orig_provider, orig_model), (res_p, res_m), accessible in zip(
                pairs, resolved_pairs, results
            ):
                if orig_provider not in _NON_VERTEX_PROVIDERS:
                    continue
                tier = _classify_openai_tier(orig_model)
                if accessible is True:
                    logger.warning(
                        f"Cloud model mapping: {orig_provider}/{orig_model} -> "
                        f"google/{res_m} ({orig_provider} has no Vertex publisher, "
                        f"tier: {tier})"
                    )
                else:
                    logger.error(
                        f"Cloud model mapping failed: {orig_provider}/{orig_model} "
                        f"-> no accessible Gemini model found (tier: {tier})"
                    )

            # Build results
            missing = [
                pair for pair, acc in zip(pairs, results) if acc is False
            ]
            verified = [
                pair for pair, acc in zip(pairs, results) if acc is True
            ]
            resolved_mapping: dict[tuple[str, str], tuple[str, str]] = {
                requested: resolved
                for requested, resolved, acc in zip(pairs, resolved_pairs, results)
                if acc is True
            }

            # For still-missing models, try the closest available names --
            # the first one that's accessible becomes the resolved substitute.
            # Skip _NON_VERTEX_PROVIDERS: they always map to Gemini (above),
            # not to their own publisher catalog.
            suggestions: dict[tuple[str, str], list[str]] = {}
            needs_terms: dict[tuple[str, str], str] = {}
            still_missing: list[tuple[str, str]] = []
            for provider, model in missing:
                if provider in _NON_VERTEX_PROVIDERS:
                    still_missing.append((provider, model))
                    continue

                top: list[str] = []
                model_in_catalog = False
                try:
                    publisher = _resolve_publisher(provider)
                    available = await self._load_publisher_models(publisher)
                    model_in_catalog = model in available
                    if available:
                        scored = [
                            (m, _model_similarity(model, m))
                            for m in available if m != model
                        ]
                        scored.sort(key=lambda x: x[1], reverse=True)
                        top = [m for m, s in scored[:3] if s >= 0.25]
                except Exception:
                    pass
                if top:
                    suggestions[(provider, model)] = top

                selected = None
                for alt in top:
                    probe = await self._probe_model_access(provider, alt)
                    if probe is True:
                        selected = alt
                        break
                    if probe is False:
                        enabled = await self._auto_enable_partner_model(provider, alt)
                        if enabled and await self._probe_model_access(provider, alt) is True:
                            selected = alt
                            break

                if selected:
                    verified.append((provider, model))
                    resolved_mapping[(provider, model)] = (provider, selected)
                    logger.warning(
                        f"Cloud model mapping: {provider}/{model} -> {provider}/{selected} "
                        f"(closest available match)"
                    )
                else:
                    if model_in_catalog:
                        needs_terms[(provider, model)] = self.model_page_url(
                            provider, model
                        )
                    still_missing.append((provider, model))

            _remember_resolved_models(self.project_id, resolved_mapping)
            if self._model_regions:
                _remember_model_regions(self.project_id, self._model_regions)
        finally:
            await self.close()

        return still_missing, verified, suggestions, needs_terms

    async def _get_partner_service_name(self, provider: str, model: str) -> Optional[str]:
        """Extract the Cloud Partner Service name for a Model Garden model.

        Fetches model metadata and parses the ``requestAccess`` URI for the
        ``service=`` parameter (e.g. ``mistral-small-2503.cloudpartnerservices.goog``).
        Returns None for Google models or if the metadata doesn't include service info.
        """
        if provider == "google":
            return None
        try:
            publisher = _resolve_publisher(provider)
            url = (
                f"https://us-central1-aiplatform.googleapis.com/v1beta1/"
                f"publishers/{publisher}/models/{model}"
            )
            token = _get_access_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "x-goog-user-project": self.project_id,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
            access_refs = (
                data.get("supportedActions", {})
                .get("requestAccess", {})
                .get("references", {})
            )
            for region_info in access_refs.values():
                uri = region_info.get("uri", "")
                for param in uri.split("&"):
                    if param.startswith("service="):
                        return param.split("=", 1)[1]
        except Exception:
            pass
        return None

    async def _auto_enable_partner_model(self, provider: str, model: str) -> bool:
        """Try to enable a partner model's Cloud Partner Service via the Service Usage API.

        Returns True if the service was enabled (or already enabled), False on failure.
        """
        service_name = await self._get_partner_service_name(provider, model)
        if not service_name:
            return False

        logger.info(f"Auto-enabling Model Garden service: {service_name}")

        url = (
            f"https://serviceusage.googleapis.com/v1/"
            f"projects/{self.project_id}/services/{service_name}:enable"
        )
        token = _get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={}, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Enabled {service_name} for project {self.project_id}")
                        await asyncio.sleep(2)
                        return True
                    else:
                        error = await response.text()
                        logger.warning(
                            f"Could not auto-enable {service_name}: "
                            f"HTTP {response.status} — {error[:200]}"
                        )
        except Exception as exc:
            logger.warning(f"Could not auto-enable {service_name}: {exc}")
        return False

    async def _probe_single_region(
        self, provider: str, model: str, region: str,
    ) -> Optional[bool]:
        """Probe a model in a specific region. Returns True/False/None."""
        request = CloudLLMRequest(
            prompt="hi",
            model=model,
            provider=provider,
            max_tokens=1,
            temperature=None,
            top_p=None,
            response_schema=None,
        )

        old_region = self.configured_region
        self.configured_region = region
        try:
            url, payload = self._build_request(request)
        finally:
            self.configured_region = old_region

        token = _get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if self._session is None:
            self._session = aiohttp.ClientSession()

        async with self._session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                return True
            if response.status == 429:
                if provider not in ("google", "anthropic"):
                    return True
                return None
            if response.status == 404:
                return False
            return None

    async def _probe_model_access(self, provider: str, model: str) -> Optional[bool]:
        """Return whether this project can call `model`.

        Tries the configured region first, then falls back to
        ``_FALLBACK_PROBE_REGIONS`` since many models are region-locked
        (e.g. Meta Llama → us-east5, Gemini 3.x → global/us/eu).

        Anthropic is the only provider that skips the fallback sweep
        because it has its own region routing and 429 can't be trusted.
        """
        primary_region = _region_for_provider(provider, self.configured_region)
        result = await self._probe_single_region(provider, model, primary_region)
        if result is True:
            return True
        if result is None:
            return None

        if provider == "anthropic":
            return result

        had_inconclusive = False
        for region in _FALLBACK_PROBE_REGIONS:
            if region == primary_region:
                continue
            alt_result = await self._probe_single_region(provider, model, region)
            if alt_result is True:
                key = _model_key(provider, model)
                self._model_regions[key] = region
                logger.info(
                    f"Model {provider}/{model} found in region {region}"
                )
                return True
            if alt_result is None:
                had_inconclusive = True

        return None if had_inconclusive else False
