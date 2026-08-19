"""Tests for Cloud LLM (VertexLLM) and cloud detection."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp

from lamia_cloud.gcp.llm.vertex import (
    VertexLLM,
    is_on_gcp,
    _get_project_id,
    _region_for_provider,
    _resolve_publisher,
    _vertex_endpoint_host,
    ANTHROPIC_UNAVAILABLE_REGIONS,
    DEFAULT_REGION,
    VERTEX_API_VERSION,
    VERTEX_REGION,
    PUBLISHER_ALIASES,
)
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse
from lamia_cloud import get_cloud_llm, is_on_cloud


class TestVertexLLMClassMethods:
    def test_is_available_delegates_to_is_on_gcp(self):
        llm = VertexLLM()
        with patch("lamia_cloud.gcp.llm.vertex.is_on_gcp", return_value=True):
            assert llm.is_available() is True
        with patch("lamia_cloud.gcp.llm.vertex.is_on_gcp", return_value=False):
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
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
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
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
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
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
    async def test_generate_mistral_model_uses_maas_rawpredict(self, mock_token, llm):
        request = CloudLLMRequest(
            prompt="test", model="mistral-small-2503", provider="mistralai", max_tokens=500
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        result = await llm.generate(request)

        assert isinstance(result, CloudLLMResponse)
        assert result.text == "Hello!"
        assert result.usage["input_tokens"] == 5
        assert result.usage["output_tokens"] == 3
        assert result.usage["total_tokens"] == 8

        url = mock_session.post.call_args[0][0]
        assert "publishers/mistralai" in url
        assert "mistral-small-2503" in url
        assert ":rawPredict" in url

        payload = mock_session.post.call_args[1]["json"]
        assert payload["model"] == "mistral-small-2503"
        assert payload["messages"] == [{"role": "user", "content": "test"}]

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
    async def test_generate_meta_model_uses_maas_rawpredict(self, mock_token, llm):
        request = CloudLLMRequest(
            prompt="test", model="llama-3.1-405b-instruct-maas", provider="meta", max_tokens=500
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        result = await llm.generate(request)

        assert result.text == "Hi there!"
        url = mock_session.post.call_args[0][0]
        assert "publishers/meta" in url
        assert ":rawPredict" in url

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
    async def test_generate_arbitrary_partner_uses_rawpredict(self, mock_token, llm):
        """Any provider not in _NON_VERTEX_PROVIDERS goes through rawPredict."""
        request = CloudLLMRequest(
            prompt="test", model="kimi-k3", provider="moonshotai", max_tokens=100
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        result = await llm.generate(request)

        assert result.text == "Hi!"
        url = mock_session.post.call_args[0][0]
        assert "publishers/moonshotai" in url
        assert "kimi-k3" in url
        assert ":rawPredict" in url

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
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
    def test_anthropic_is_itself(self):
        assert _resolve_publisher("anthropic") == "anthropic"

    def test_google_is_itself(self):
        assert _resolve_publisher("google") == "google"

    def test_mistralai_is_itself(self):
        assert _resolve_publisher("mistralai") == "mistralai"

    def test_meta_is_itself(self):
        assert _resolve_publisher("meta") == "meta"

    def test_unknown_provider_is_itself(self):
        assert _resolve_publisher("moonshotai") == "moonshotai"

    def test_mistral_alias_maps_to_mistralai(self):
        assert _resolve_publisher("mistral") == "mistralai"


class TestRegionForProvider:
    def test_anthropic_at_unavailable_region_upgrades_to_global(self):
        for region in ANTHROPIC_UNAVAILABLE_REGIONS:
            assert _region_for_provider("anthropic", region) == "global"

    def test_anthropic_explicit_region_used_as_is(self):
        assert _region_for_provider("anthropic", "us-east5") == "us-east5"

    def test_anthropic_explicit_multi_region_used_as_is(self):
        assert _region_for_provider("anthropic", "eu") == "eu"

    def test_anthropic_explicit_global_used_as_is(self):
        assert _region_for_provider("anthropic", "global") == "global"

    def test_google_region_never_rewritten(self):
        assert _region_for_provider("google", DEFAULT_REGION) == DEFAULT_REGION
        assert _region_for_provider("google", "us-east5") == "us-east5"


class TestVertexEndpointHost:
    def test_global(self):
        assert _vertex_endpoint_host("global") == "aiplatform.googleapis.com"

    def test_multi_region_us(self):
        assert _vertex_endpoint_host("us") == "aiplatform.us.rep.googleapis.com"

    def test_multi_region_eu(self):
        assert _vertex_endpoint_host("eu") == "aiplatform.eu.rep.googleapis.com"

    def test_specific_region(self):
        assert _vertex_endpoint_host("us-east5") == "us-east5-aiplatform.googleapis.com"


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

    @patch("lamia_cloud.gcp.llm.vertex.is_on_gcp", return_value=False)
    def test_is_on_cloud_false_locally(self, mock):
        assert is_on_cloud() is False


class TestProbeModelAccess:
    """Three states, not two. True/False only for a status we have direct,
    confirmed evidence for (200 / the specific 404 "not found or no access"
    shape). Everything else -- including 429 -- is None (inconclusive): a
    quota/rate-limit error names a real, permitted model and fires
    regardless of Model Garden consent, so it proves nothing about access
    either way and must not be reported as either accessible or inaccessible."""

    @pytest.fixture
    def llm(self):
        instance = VertexLLM()
        instance.project_id = "test-project-id"
        return instance

    async def _probe_with_status(self, llm, status: int, body: str = ""):
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.text = AsyncMock(return_value=body)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session
        with patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token"):
            return await llm._probe_model_access("anthropic", "claude-sonnet-4-5-20250929")

    @pytest.mark.asyncio
    async def test_200_is_accessible(self, llm):
        assert await self._probe_with_status(llm, 200) is True

    @pytest.mark.asyncio
    async def test_404_not_found_is_inaccessible(self, llm):
        body = '{"error": {"code": 404, "message": "...was not found or your project does not have access to it..."}}'
        assert await self._probe_with_status(llm, 404, body) is False

    @pytest.mark.asyncio
    async def test_404_without_recognized_phrase_is_inconclusive(self, llm):
        # A different 404 error shape isn't the confirmed consent-denial
        # signal -- must not be silently treated as accessible or inaccessible.
        assert await self._probe_with_status(llm, 404, '{"error": {"message": "unrelated"}}') is None

    @pytest.mark.asyncio
    async def test_403_is_inconclusive(self, llm):
        assert await self._probe_with_status(llm, 403, '{"error": {"message": "forbidden"}}') is None

    @pytest.mark.asyncio
    async def test_400_is_inconclusive(self, llm):
        assert await self._probe_with_status(llm, 400, '{"error": {"message": "bad request"}}') is None

    @pytest.mark.asyncio
    async def test_429_quota_exceeded_is_inconclusive(self, llm):
        # Confirmed via live testing: a genuinely disabled model (no Model
        # Garden consent) returns this exact same 429 shape, so it cannot be
        # treated as proof of access.
        body = '{"error": {"code": 429, "message": "Quota exceeded for ... base model: anthropic-claude-haiku-4-5", "status": "RESOURCE_EXHAUSTED"}}'
        assert await self._probe_with_status(llm, 429, body) is None


class TestCheckModelAccess:
    """check_model_access must split a batch into exactly the pairs proven
    accessible and exactly the pairs proven inaccessible -- a pair whose
    probe was inconclusive (None) must land in neither list, not get
    silently folded into one side or the other."""

    @pytest.fixture
    def llm(self):
        instance = VertexLLM()
        instance.project_id = "test-project-id"
        return instance

    def test_mixed_batch_splits_correctly(self, llm):
        results = {
            "claude-sonnet-4-5-20250929": True,   # accessible
            "claude-opus-4-5-20251101": False,    # confirmed inaccessible
            "claude-haiku-4-5-20251001": None,    # inconclusive (e.g. 429)
        }

        async def fake_select(model):
            return model

        async def fake_probe(provider, model):
            return results[model]

        llm._select_anthropic_model = fake_select
        llm._probe_model_access = fake_probe

        missing, verified = llm.check_model_access([
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("anthropic", "claude-opus-4-5-20251101"),
            ("anthropic", "claude-haiku-4-5-20251001"),
        ])

        assert missing == [("anthropic", "claude-opus-4-5-20251101")]
        assert verified == [("anthropic", "claude-sonnet-4-5-20250929")]
        # The inconclusive pair must appear in neither list.
        for pair_list in (missing, verified):
            assert ("anthropic", "claude-haiku-4-5-20251001") not in pair_list

    def test_google_passes_through_as_verified_untested(self, llm):
        missing, verified = llm.check_model_access([("google", "gemini-3.5-flash")])
        assert missing == []
        assert verified == [("google", "gemini-3.5-flash")]

    def test_any_non_google_provider_is_gated(self, llm):
        """All non-Google providers are gated by Model Garden."""
        async def fake_probe(provider, model):
            return True

        llm._probe_model_access = fake_probe

        missing, verified = llm.check_model_access([
            ("mistralai", "mistral-small-2503"),
            ("moonshotai", "kimi-k3"),
        ])
        assert missing == []
        assert ("mistralai", "mistral-small-2503") in verified
        assert ("moonshotai", "kimi-k3") in verified

    def test_empty_input_returns_empty_lists(self, llm):
        assert llm.check_model_access([]) == ([], [])
