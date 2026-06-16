"""GCP Vertex AI service implementation.

Routes LLM calls through Vertex AI using ADC. On Cloud Run, authentication
is automatic via the metadata server. Locally, uses `gcloud auth application-default login`.

Supports two publisher types:
- Anthropic (claude-*) → rawPredict endpoint with anthropic_version header
- Google (gemini-*) → generateContent endpoint with Gemini format
"""
import logging
import os
import re
import urllib.request
from typing import Optional, Dict, Any

import aiohttp

from lamia_cloud.interfaces import CloudLLM
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse

logger = logging.getLogger(__name__)

ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
DEFAULT_REGION = "us-central1"
ANTHROPIC_REGIONS = ("us-east5", "europe-west1")
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

# Backward-compatible exports used by existing tests/integrations.
VERTEX_API_VERSION = ANTHROPIC_VERTEX_VERSION
VERTEX_REGION = DEFAULT_REGION


def _resolve_publisher(provider: str) -> str:
    """Compatibility helper: map non-anthropic providers to google."""
    return "anthropic" if provider == "anthropic" else "google"


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
    """Resolve the region based on provider. Anthropic needs specific regions."""
    if provider == "anthropic":
        if configured_region in ANTHROPIC_REGIONS:
            return configured_region
        return ANTHROPIC_REGIONS[0]
    return configured_region


class VertexLLM(CloudLLM):
    """Cloud LLM backed by GCP Vertex AI.

    Automatically routes to the correct endpoint based on provider:
    - anthropic → rawPredict (with anthropic_version)
    - google → generateContent (Gemini format)
    """

    def __init__(self, region: str = DEFAULT_REGION) -> None:
        self.project_id = _get_project_id()
        self.configured_region = region
        # Keep old public attribute name for compatibility.
        self.region = region
        self._session: Optional[aiohttp.ClientSession] = None
        self._google_models_cache: list[str] | None = None
        self._anthropic_version = ANTHROPIC_VERTEX_VERSION

    def is_available(self) -> bool:
        return is_on_gcp()

    def _build_anthropic_request(
        self,
        request: CloudLLMRequest,
        anthropic_version: str,
    ) -> tuple[str, Dict[str, Any]]:
        """Build rawPredict request for Anthropic models on Vertex."""
        region = _region_for_provider("anthropic", self.configured_region)
        url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{region}/"
            f"publishers/anthropic/models/{request.model}:rawPredict"
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

    def _build_google_request(self, request: CloudLLMRequest) -> tuple[str, Dict[str, Any]]:
        """Build generateContent request for Google models (Gemini) on Vertex."""
        region = _region_for_provider("google", self.configured_region)
        url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
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

    async def _load_google_models(self) -> list[str]:
        """List available Google publisher models from Vertex AI."""
        if self._google_models_cache is not None:
            return self._google_models_cache

        region = _region_for_provider("google", self.configured_region)
        url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{region}/publishers/google/models"
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

        self._google_models_cache = sorted(set(models))
        return self._google_models_cache

    async def _select_google_model_for_tier(self, tier: str) -> str:
        """Select best available Gemini model for the requested tier."""
        available = await self._load_google_models()

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

    def _resolve_model(self, request: CloudLLMRequest) -> CloudLLMRequest:
        """Map non-native providers to a Gemini tier placeholder.

        Anthropic and Google are natively supported on Vertex and are left as-is.
        Other providers (OpenAI/OpenRouter/etc.) are mapped by capability tier.
        """
        if request.provider in ("google", "anthropic"):
            return request

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

    @staticmethod
    def _extract_anthropic_version_hint(error_text: str) -> str | None:
        """Extract a newer anthropic_version from API error text, if present."""
        versions = re.findall(r"vertex-\d{4}-\d{2}-\d{2}", error_text)
        return versions[-1] if versions else None

    async def generate(self, request: CloudLLMRequest) -> CloudLLMResponse:
        original_provider = request.provider
        original_model = request.model
        request = self._resolve_model(request)
        is_anthropic = request.provider == "anthropic"

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

        if is_anthropic:
            url, payload = self._build_anthropic_request(request, self._anthropic_version)
        else:
            url, payload = self._build_google_request(request)

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
                if is_anthropic:
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
                            self._handle_api_error(retry_response.status, error_text)
                            raise RuntimeError(
                                f"Vertex AI error ({retry_response.status}): {error_text}"
                            )

                self._handle_api_error(response.status, error_text)
                raise RuntimeError(f"Vertex AI error ({response.status}): {error_text}")

            data = await response.json()

            if is_anthropic:
                return self._parse_anthropic_response(data, request.model)
            return self._parse_google_response(data, request.model)

    def _handle_api_error(self, status: int, error_text: str) -> None:
        """Log actionable guidance for common Vertex AI errors."""
        if status == 404 and "was not found" in error_text:
            tos_url = (
                f"https://console.cloud.google.com/vertex-ai"
                f"?project={self.project_id}"
            )
            logger.error(
                f"Model not accessible. Accept Vertex AI terms of service at:\n"
                f"  {tos_url}\n"
                f"Then retry. If running locally, also try:\n"
                f"  gcloud services enable aiplatform.googleapis.com "
                f"--project={self.project_id}"
            )
            try:
                import webbrowser
                webbrowser.open(tos_url)
            except Exception:
                pass
        elif status == 403 and "SERVICE_DISABLED" in error_text:
            logger.error(
                f"Vertex AI API not enabled. Enabling automatically..."
            )

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
