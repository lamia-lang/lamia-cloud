"""Tests for Cloud LLM (VertexLLM) and cloud detection."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp

from lamia_cloud.gcp.vertex import (
    VertexLLM,
    is_on_gcp,
    _get_project_id,
    _resolve_publisher,
    VERTEX_API_VERSION,
    VERTEX_REGION,
)
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse
from lamia_cloud import get_cloud_llm, is_on_cloud


class TestVertexLLMClassMethods:
    def test_is_available_delegates_to_is_on_gcp(self):
        llm = VertexLLM()
        with patch("lamia_cloud.gcp.vertex.is_on_gcp", return_value=True):
            assert llm.is_available() is True
        with patch("lamia_cloud.gcp.vertex.is_on_gcp", return_value=False):
            assert llm.is_available() is False


class TestVertexLLMInit:
    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-project-123"})
    def test_init_from_env_var(self):
        llm = VertexLLM()
        assert llm.project_id == "my-project-123"
        assert llm.region == VERTEX_REGION


class TestVertexLLMGenerate:
    @pytest.fixture
    def llm(self):
        instance = VertexLLM()
        instance.project_id = "test-project-id"
        return instance

    @pytest.fixture
    def llm_request(self):
        return CloudLLMRequest(
            prompt="Hello",
            model="claude-sonnet-4-6",
            provider="anthropic",
            max_tokens=1024,
        )

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.vertex._get_access_token", return_value="fake-token")
    async def test_generate_success(self, mock_token, llm, llm_request):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"text": "Hello from Vertex!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        result = await llm.generate(llm_request)

        assert isinstance(result, CloudLLMResponse)
        assert result.text == "Hello from Vertex!"
        assert result.usage["total_tokens"] == 15

        call_args = mock_session.post.call_args
        url = call_args[0][0]
        assert "test-project-id" in url
        assert "publishers/anthropic" in url
        assert "claude-sonnet-4-6" in url
        assert ":rawPredict" in url

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.vertex._get_access_token", return_value="fake-token")
    async def test_generate_openai_model_uses_google_publisher(self, mock_token, llm):
        request = CloudLLMRequest(prompt="test", model="gpt-4o", provider="openai", max_tokens=500)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"text": "response"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        await llm.generate(request)

        url = mock_session.post.call_args[0][0]
        assert "publishers/google" in url

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.vertex._get_access_token", return_value="fake-token")
    async def test_generate_api_error(self, mock_token, llm, llm_request):
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        with pytest.raises(RuntimeError, match="Vertex AI error"):
            await llm.generate(llm_request)


class TestCloudDetection:
    @patch("urllib.request.urlopen")
    def test_is_on_gcp_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value = mock_resp
        assert is_on_gcp() is True

    @patch("urllib.request.urlopen", side_effect=Exception("connection refused"))
    def test_is_on_gcp_false(self, mock_urlopen):
        assert is_on_gcp() is False

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-project-id"})
    def test_get_project_id_from_env(self):
        assert _get_project_id() == "env-project-id"

    @patch.dict("os.environ", {}, clear=True)
    @patch("urllib.request.urlopen")
    def test_get_project_id_from_metadata(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"metadata-project-id"
        mock_urlopen.return_value = mock_resp
        assert _get_project_id() == "metadata-project-id"


class TestPublisherResolution:
    def test_anthropic_publisher(self):
        assert _resolve_publisher("anthropic") == "anthropic"

    def test_openai_maps_to_google(self):
        assert _resolve_publisher("openai") == "google"

    def test_unknown_maps_to_google(self):
        assert _resolve_publisher("some_provider") == "google"


class TestVertexLLMClose:
    @pytest.mark.asyncio
    async def test_close_with_session(self):
        llm = VertexLLM()
        mock_session = AsyncMock()
        llm._session = mock_session
        await llm.close()
        mock_session.close.assert_called_once()
        assert llm._session is None

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        llm = VertexLLM()
        llm._session = None
        await llm.close()


class TestPublicAPI:
    def test_get_cloud_llm_returns_vertex(self):
        llm = get_cloud_llm()
        assert isinstance(llm, VertexLLM)

    @patch("lamia_cloud.gcp.vertex.is_on_gcp", return_value=False)
    def test_is_on_cloud_false_locally(self, mock):
        assert is_on_cloud() is False
