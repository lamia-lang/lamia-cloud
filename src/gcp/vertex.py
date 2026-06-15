"""GCP Vertex AI service implementation.

Routes LLM calls through Vertex AI using ADC. On Cloud Run, authentication
is automatic via the metadata server. Locally, uses `gcloud auth application-default login`.
"""
import logging
import os
import urllib.request
from typing import Optional, Dict, Any

import aiohttp

from lamia_cloud.interfaces import CloudLLM
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse

logger = logging.getLogger(__name__)

VERTEX_API_VERSION = "vertex-2023-10-16"
VERTEX_REGION = "global"


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


def _resolve_publisher(provider: str) -> str:
    """Map provider name to Vertex AI publisher."""
    if provider == "anthropic":
        return "anthropic"
    return "google"


class VertexLLM(CloudLLM):
    """Cloud LLM backed by GCP Vertex AI."""

    def __init__(self) -> None:
        self.project_id = _get_project_id()
        self.region = VERTEX_REGION
        self._session: Optional[aiohttp.ClientSession] = None

    def is_available(self) -> bool:
        return is_on_gcp()

    async def generate(self, request: CloudLLMRequest) -> CloudLLMResponse:
        publisher = _resolve_publisher(request.provider)
        url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/"
            f"publishers/{publisher}/models/{request.model}:rawPredict"
        )

        payload: Dict[str, Any] = {
            "anthropic_version": VERTEX_API_VERSION,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
        }

        if request.top_p is not None and request.temperature is None:
            payload["top_p"] = request.top_p
        else:
            payload["temperature"] = request.temperature if request.temperature is not None else 0.7

        if request.response_schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": request.response_schema}
            }

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
                raise RuntimeError(f"Vertex AI error ({response.status}): {error_text}")

            data = await response.json()

            return CloudLLMResponse(
                text=data["content"][0]["text"],
                model=request.model,
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

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
