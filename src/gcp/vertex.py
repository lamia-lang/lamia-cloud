"""GCP Vertex AI service implementation.

Routes LLM calls through Vertex AI using ADC. On Cloud Run, authentication
is automatic via the metadata server. Locally, uses `gcloud auth application-default login`.

Supports two publisher types:
- Anthropic (claude-*) → rawPredict endpoint with anthropic_version header
- Google (gemini-*) → generateContent endpoint with Gemini format
"""
import logging
import os
import urllib.request
from typing import Optional, Dict, Any

import aiohttp

from lamia_cloud.interfaces import CloudLLM
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse

logger = logging.getLogger(__name__)

ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
DEFAULT_REGION = "us-central1"
ANTHROPIC_REGIONS = ("us-east5", "europe-west1")

OPENAI_TO_VERTEX_MAP = {
    "gpt-4o": "gemini-2.0-flash",
    "gpt-4o-mini": "gemini-2.0-flash",
    "gpt-4-turbo": "gemini-2.0-flash",
    "gpt-4": "gemini-1.5-pro",
    "gpt-3.5-turbo": "gemini-2.0-flash",
    "o1": "gemini-2.0-flash",
    "o1-mini": "gemini-2.0-flash",
    "o1-preview": "gemini-1.5-pro",
}


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
        self._session: Optional[aiohttp.ClientSession] = None

    def is_available(self) -> bool:
        return is_on_gcp()

    def _build_anthropic_request(self, request: CloudLLMRequest) -> tuple[str, Dict[str, Any]]:
        """Build rawPredict request for Anthropic models on Vertex."""
        region = _region_for_provider("anthropic", self.configured_region)
        url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{region}/"
            f"publishers/anthropic/models/{request.model}:rawPredict"
        )

        payload: Dict[str, Any] = {
            "anthropic_version": ANTHROPIC_VERTEX_VERSION,
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
        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        usage_meta = data.get("usageMetadata", {})
        return CloudLLMResponse(
            text=text,
            model=model,
            usage={
                "input_tokens": usage_meta.get("promptTokenCount", 0),
                "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            },
            raw=data,
        )

    def _resolve_model(self, request: CloudLLMRequest) -> CloudLLMRequest:
        """Map OpenAI models to Vertex AI equivalents transparently."""
        if request.provider == "openai" and request.model in OPENAI_TO_VERTEX_MAP:
            mapped = OPENAI_TO_VERTEX_MAP[request.model]
            logger.info(
                f"Cloud model mapping: {request.provider}/{request.model} → google/{mapped}"
            )
            return CloudLLMRequest(
                prompt=request.prompt,
                model=mapped,
                provider="google",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                response_schema=request.response_schema,
            )
        return request

    async def generate(self, request: CloudLLMRequest) -> CloudLLMResponse:
        request = self._resolve_model(request)
        is_anthropic = request.provider == "anthropic"

        if is_anthropic:
            url, payload = self._build_anthropic_request(request)
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
