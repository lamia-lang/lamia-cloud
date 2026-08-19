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
import logging
import os
import re
import urllib.request
from datetime import date
from typing import Optional, Dict, Any

import aiohttp
from google.cloud import resourcemanager_v3

from lamia_cloud.gcp.llm import anthropic_mapper
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
    "strong": "gemini-3.1-pro-preview",
    "medium": "gemini-3.5-flash",
    "light": "gemini-3.1-flash-lite",
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


def get_verified_vertex_models(project_id: str) -> set[tuple[str, str]]:
    """(provider, model) pairs already confirmed accessible on Vertex AI for this project.

    Stored as labels on the GCP project itself, so the cache is shared across
    every team member and CI run against that project, not just one checkout.
    """
    try:
        client = resourcemanager_v3.ProjectsClient()
        project = client.get_project(name=f"projects/{project_id}")
        labels = project.labels or {}
    except Exception:
        return set()
    pairs = set()
    for key in labels:
        if not key.startswith(_VERTEX_MODEL_LABEL_PREFIX):
            continue
        provider, _, model = key[len(_VERTEX_MODEL_LABEL_PREFIX):].partition("-")
        if provider and model:
            pairs.add((provider, model))
    return pairs


def remember_verified_vertex_models(project_id: str, models: set[tuple[str, str]]) -> None:
    """Label the project with confirmed Vertex AI model access.

    Provider and model ids are already label-safe (lowercase letters, digits,
    hyphens); any pair whose key doesn't fit the 63-char label key limit is
    skipped rather than failing the whole call.
    """
    try:
        client = resourcemanager_v3.ProjectsClient()
        project = client.get_project(name=f"projects/{project_id}")
        if project.labels is None:
            project.labels = {}
        today = date.today().strftime("%Y%m%d")
        changed = False
        for provider, model in models:
            key = _vertex_model_label_key(provider, model)
            if len(key) > 63:
                continue
            if project.labels.get(key) != today:
                project.labels[key] = today
                changed = True
        if changed:
            client.update_project(project=project)
    except Exception as exc:
        logger.warning(f"Failed to cache verified Vertex AI models on project {project_id}: {exc}")


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


_VERTEX_MODEL_LABEL_PREFIX = "lamia-vertex-model-"


def _vertex_model_label_key(provider: str, model: str) -> str:
    return f"{_VERTEX_MODEL_LABEL_PREFIX}{provider}-{model}"


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
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """All non-Google models are gated by Vertex AI Model Garden.

        A pair whose live check comes back inconclusive (e.g. rate-limited)
        appears in neither returned list.
        """
        gated: list[tuple[str, str]] = []
        passthrough: list[tuple[str, str]] = []
        for provider, model in models:
            if provider == "google":
                passthrough.append((provider, model))
            else:
                gated.append((provider, model))
        if not gated:
            return [], passthrough
        missing, verified = asyncio.run(self._check_gated_access(gated))
        return missing, verified + passthrough

    def catalog_display_name(self, provider: str, model: str) -> str:
        if provider != "anthropic":
            return model
        return anthropic_mapper.model_garden_name(model)

    def model_catalog_url(self) -> str:
        return f"https://console.cloud.google.com/agent-platform/model-garden?project={self.project_id}"

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

        region = _region_for_provider(publisher, self.configured_region)
        url = (
            f"https://{_vertex_endpoint_host(region)}/v1/"
            f"projects/{self.project_id}/locations/{region}/publishers/{publisher}/models"
        )
        token = _get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        models: list[str] = []
        try:
            # Use a dedicated short-lived session so model discovery does not
            # interfere with the request session/mocks used by generate().
            async with aiohttp.ClientSession() as discovery_session:
                async with discovery_session.get(url, headers=headers) as response:
                    if response.status == 200:
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
        except Exception:
            pass

        result = sorted(set(models))
        self._publisher_models_cache[publisher] = result
        return result

    async def _select_google_model_for_tier(self, tier: str) -> str:
        """Select best available Gemini model for the requested tier."""
        available = await self._load_publisher_models("google")

        if available:
            if tier == "strong":
                candidates = [
                    m for m in available
                    if m.startswith("gemini-") and "pro" in m and "preview" not in m
                ]
                if not candidates:
                    candidates = [m for m in available if m.startswith("gemini-") and "pro" in m]
            elif tier == "light":
                candidates = [
                    m for m in available
                    if m.startswith("gemini-") and ("flash-lite" in m or "lite" in m)
                ]
            else:
                candidates = [
                    m for m in available
                    if m.startswith("gemini-") and "flash" in m and "lite" not in m
                ]

            if candidates:
                candidates.sort(key=_extract_version_score, reverse=True)
                return candidates[0]

        return FALLBACK_GEMINI_MODELS[tier]

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
        region = _region_for_provider(request.provider, self.configured_region)
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
        region = _region_for_provider("google", self.configured_region)
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
            text = candidate["content"]["parts"][0]["text"]
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
        if status == 404 and "was not found" in error_text:
            model_garden_url = self.model_catalog_url()
            search_hint = self.catalog_display_name(request.provider, request.model)
            logger.error(
                f"Model not accessible. In Model Garden, search \"{search_hint}\" "
                f"and click Enable:\n"
                f"  {model_garden_url}\n"
                f"Then retry."
            )
            try:
                import webbrowser
                webbrowser.open(model_garden_url)
            except Exception:
                pass
        elif status == 403 and "SERVICE_DISABLED" in error_text:
            logger.error(
                f"Vertex AI API not enabled. Enabling automatically..."
            )

    async def _check_gated_access(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Check Model Garden access for Anthropic and MaaS partner models."""
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
                else:
                    resolved_pairs.append((provider, model))
            results = await asyncio.gather(
                *(self._probe_model_access(provider, model) for provider, model in resolved_pairs)
            )
        finally:
            await self.close()
        missing = [pair for pair, accessible in zip(resolved_pairs, results) if accessible is False]
        verified = [pair for pair, accessible in zip(resolved_pairs, results) if accessible is True]
        return missing, verified

    async def _probe_model_access(self, provider: str, model: str) -> Optional[bool]:
        """Return whether this project can call `model`, without logging or opening a browser.

        Sends one minimal live request and reads the status. True/False only
        for a status we have direct, confirmed evidence for (200 / the
        specific 404 "not found or no access" shape); None for anything else.
        """
        request = CloudLLMRequest(
            prompt="hi",
            model=model,
            provider=provider,
            max_tokens=1,
            temperature=None,
            top_p=None,
            response_schema=None,
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
            if response.status == 200:
                return True
            if response.status == 404:
                error_text = await response.text()
                if "was not found" in error_text:
                    return False
            return None
