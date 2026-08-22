"""Tests for Cloud LLM (VertexLLM) and cloud detection."""

import json

import pytest
from google.api_core.exceptions import PreconditionFailed
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp

from lamia_cloud.gcp.llm.vertex import (
    VertexLLM,
    is_on_gcp,
    get_verified_vertex_models,
    remember_verified_vertex_models,
    _get_cached_resolution,
    _remember_resolved_models,
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


@pytest.fixture(autouse=True)
def no_real_gcs_by_default(monkeypatch):
    """Tests unrelated to the verified-models cache still exercise code
    paths (_get_cached_resolution, _remember_resolved_models) that call
    storage.Client(). Fail fast by default instead of a real network call;
    tests that actually cover the cache set up their own mock, which
    overrides this since it runs later."""
    def raise_immediately(project):
        raise RuntimeError("no storage client in tests")
    monkeypatch.setattr("lamia_cloud.gcp.llm.vertex.storage.Client", raise_immediately)


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
            model="claude-sonnet-4-5-20250514",
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
        assert "claude-sonnet-4-5@20250514" in url
        assert ":rawPredict" in url

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
    @patch("lamia_cloud.gcp.llm.vertex._get_cached_resolution")
    async def test_generate_applies_cached_anthropic_resolution(self, mock_cache, mock_token, llm):
        """User writes claude-sonnet-4-5 in config; pre-deploy resolved it to
        the dated Model Garden version claude-sonnet-4-5-20250514. At runtime,
        generate() must apply the cached mapping."""
        mock_cache.return_value = ("anthropic", "claude-sonnet-4-5-20250514")
        request = CloudLLMRequest(
            prompt="Hello", model="claude-sonnet-4-5", provider="anthropic", max_tokens=1024,
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        await llm.generate(request)

        mock_cache.assert_called_once_with("test-project-id", "anthropic", "claude-sonnet-4-5")
        url = mock_session.post.call_args[0][0]
        assert "claude-sonnet-4-5@20250514" in url
        assert "claude-sonnet-4-5/" not in url

    @pytest.mark.asyncio
    @patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token")
    @patch("lamia_cloud.gcp.llm.vertex._get_cached_resolution")
    async def test_generate_applies_cached_partner_resolution(self, mock_cache, mock_token, llm):
        """User writes mistral-small-2503 in config; pre-deploy found it
        inaccessible and auto-mapped to mistral-large-2411. At runtime,
        generate() must apply the cached mapping for non-anthropic too."""
        mock_cache.return_value = ("mistralai", "mistral-large-2411")
        request = CloudLLMRequest(
            prompt="Hello", model="mistral-small-2503", provider="mistralai", max_tokens=1024,
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session

        await llm.generate(request)

        mock_cache.assert_called_once_with("test-project-id", "mistralai", "mistral-small-2503")
        url = mock_session.post.call_args[0][0]
        assert "mistral-large-2411" in url
        assert "publishers/mistralai" in url

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

    @pytest.mark.asyncio
    async def test_429_for_partner_model_is_accessible(self, llm):
        """Partner models (meta, mistral, etc.) treat 429 as proof of access —
        unlike Anthropic, disabled partner models return 404 not 429."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value='{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        llm._session = mock_session
        with patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token"):
            assert await llm._probe_model_access("meta", "llama-4-maverick-17b-128e-instruct-maas") is True

    @pytest.mark.asyncio
    async def test_multi_region_probe_finds_model_in_alternate_region(self, llm):
        """When primary region returns 404, try alternate regions."""
        call_count = 0
        def make_response(status, body=""):
            r = AsyncMock()
            r.status = status
            r.text = AsyncMock(return_value=body)
            r.__aenter__ = AsyncMock(return_value=r)
            r.__aexit__ = AsyncMock(return_value=False)
            return r

        responses = []
        # Primary region → 404
        responses.append(make_response(404, '{"error": {"message": "was not found"}}'))
        # First alternate (us-east5 is first in PARTNER_PROBE_REGIONS) → 200
        responses.append(make_response(200))

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=responses)
        llm._session = mock_session
        with patch("lamia_cloud.gcp.llm.vertex._get_access_token", return_value="fake-token"):
            result = await llm._probe_model_access("meta", "llama-4-maverick-17b-128e-instruct-maas")
        assert result is True
        assert llm._model_regions.get("meta:llama-4-maverick-17b-128e-instruct-maas") == "us-east5"


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

        missing, verified, suggestions, _ = llm.check_model_access([
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("anthropic", "claude-opus-4-5-20251101"),
            ("anthropic", "claude-haiku-4-5-20251001"),
        ])

        assert missing == [("anthropic", "claude-opus-4-5-20251101")]
        assert verified == [("anthropic", "claude-sonnet-4-5-20250929")]
        for pair_list in (missing, verified):
            assert ("anthropic", "claude-haiku-4-5-20251001") not in pair_list

    def test_verified_keyed_by_requested_model_when_substituted(self, llm):
        """User writes claude-sonnet-4-5 (no date); Vertex catalog has
        claude-sonnet-4-5-20250514. `verified` must list the original lamia
        name, not the dated Model Garden substitute."""
        async def fake_select(model):
            if model == "claude-sonnet-4-5":
                return "claude-sonnet-4-5-20250514"
            return model

        async def fake_probe(provider, model):
            return True

        llm._select_anthropic_model = fake_select
        llm._probe_model_access = fake_probe

        missing, verified, _, _ = llm.check_model_access([("anthropic", "claude-sonnet-4-5")])

        assert missing == []
        assert verified == [("anthropic", "claude-sonnet-4-5")]

    def test_verified_keyed_by_requested_model_when_substitute_is_a_different_family(self, llm):
        """User writes claude-opus-4 but only claude-sonnet-4-5-20250514 is
        in the catalog. The family-substring fallback selects it. `verified`
        must still list the pair as the user requested it."""
        async def fake_select(model):
            if model == "claude-opus-4":
                return "claude-sonnet-4-5-20250514"
            return model

        async def fake_probe(provider, model):
            return True

        llm._select_anthropic_model = fake_select
        llm._probe_model_access = fake_probe

        missing, verified, _, _ = llm.check_model_access([("anthropic", "claude-opus-4")])

        assert missing == []
        assert verified == [("anthropic", "claude-opus-4")]

    def test_google_is_probed_like_any_other_provider(self, llm):
        async def fake_probe(provider, model):
            return True

        llm._probe_model_access = fake_probe

        missing, verified, _, _ = llm.check_model_access([("google", "gemini-2.5-flash")])
        assert missing == []
        assert verified == [("google", "gemini-2.5-flash")]

    def test_all_providers_are_probed(self, llm):
        async def fake_probe(provider, model):
            return True

        llm._probe_model_access = fake_probe

        missing, verified, _, _ = llm.check_model_access([
            ("google", "gemini-2.5-flash"),
            ("mistralai", "mistral-small-2503"),
            ("moonshotai", "kimi-k3"),
        ])
        assert missing == []
        assert ("google", "gemini-2.5-flash") in verified
        assert ("mistralai", "mistral-small-2503") in verified
        assert ("moonshotai", "kimi-k3") in verified

    def test_missing_model_auto_mapped_to_closest_available(self, llm):
        """mistral-small-2503 is not accessible on Vertex, but
        mistral-large-2411 is. check_model_access must auto-select it,
        move the pair from missing to verified, and record the suggestion."""
        probed = {}

        async def fake_probe(provider, model):
            return probed.get(model)

        async def fake_load(publisher):
            return ["mistral-small-2501", "mistral-large-2411"]

        probed["mistral-small-2503"] = False
        probed["mistral-small-2501"] = False
        probed["mistral-large-2411"] = True

        llm._probe_model_access = fake_probe
        llm._load_publisher_models = fake_load
        llm._auto_enable_partner_model = AsyncMock(return_value=False)

        missing, verified, suggestions, _ = llm.check_model_access([
            ("mistralai", "mistral-small-2503"),
        ])

        assert missing == []
        assert ("mistralai", "mistral-small-2503") in verified
        assert ("mistralai", "mistral-small-2503") in suggestions

    def test_openai_model_resolved_to_gemini_tier(self, llm):
        """openai:gpt-4 has no Vertex publisher -- check_model_access must
        map it to a Google Gemini model (medium tier) and verify that."""
        async def fake_probe(provider, model):
            if provider == "google":
                return True
            return False

        async def fake_ranked(tier):
            return ["gemini-2.5-flash"]

        llm._probe_model_access = fake_probe
        llm._ranked_google_candidates = fake_ranked

        missing, verified, _, _ = llm.check_model_access([("openai", "gpt-4")])

        assert missing == []
        assert ("openai", "gpt-4") in verified

    def test_model_in_catalog_but_inaccessible_lands_in_needs_terms(self, llm):
        """meta:llama-3.3-70b-instruct-maas exists in the Model Garden
        catalog but the probe returns 404 (requires EULA acceptance).
        It must appear in both missing and needs_terms with a direct
        Model Garden page URL."""
        async def fake_probe(provider, model):
            return False

        async def fake_load(publisher):
            return ["llama-3.3-70b-instruct-maas", "llama-3.1-8b-instruct-maas"]

        llm._probe_model_access = fake_probe
        llm._load_publisher_models = fake_load
        llm._auto_enable_partner_model = AsyncMock(return_value=False)

        missing, verified, suggestions, needs_terms = llm.check_model_access([
            ("meta", "llama-3.3-70b-instruct-maas"),
        ])

        assert ("meta", "llama-3.3-70b-instruct-maas") in missing
        assert ("meta", "llama-3.3-70b-instruct-maas") in needs_terms
        assert "publishers/meta/model-garden/llama-3.3-70b-instruct-maas" in needs_terms[("meta", "llama-3.3-70b-instruct-maas")]

    def test_model_not_in_catalog_has_no_needs_terms_entry(self, llm):
        """A typo model that doesn't exist in the catalog at all should
        NOT appear in needs_terms -- it's a wrong name, not a EULA issue."""
        async def fake_probe(provider, model):
            return False

        async def fake_load(publisher):
            return ["llama-3.3-70b-instruct-maas"]

        llm._probe_model_access = fake_probe
        llm._load_publisher_models = fake_load
        llm._auto_enable_partner_model = AsyncMock(return_value=False)

        missing, _, _, needs_terms = llm.check_model_access([
            ("meta", "llama-4-nonexistent-model"),
        ])

        assert ("meta", "llama-4-nonexistent-model") in missing
        assert ("meta", "llama-4-nonexistent-model") not in needs_terms

    def test_empty_input_returns_empty_lists(self, llm):
        assert llm.check_model_access([]) == ([], [], {}, {})


class TestVerifiedModelsCache:
    @staticmethod
    def _mock_storage(monkeypatch, existing: dict[str, str] | None, generation: int = 1):
        blob = MagicMock()
        blob.exists.return_value = existing is not None
        blob.generation = generation
        if existing is not None:
            blob.download_as_bytes.return_value = json.dumps(
                {"lamia-vertex-model-map": existing}
            ).encode()

        bucket = MagicMock()
        bucket.exists.return_value = True
        bucket.blob.return_value = blob

        client = MagicMock()
        client.bucket.return_value = bucket
        client.create_bucket.return_value = bucket

        monkeypatch.setattr(
            "lamia_cloud.gcp.llm.vertex.storage.Client", lambda project: client
        )
        return client, bucket, blob

    def test_get_verified_models_returns_pairs(self, monkeypatch):
        self._mock_storage(monkeypatch, {
            "anthropic:claude-sonnet-4-5-20250514": "anthropic:claude-sonnet-4-5-20250514",
        })

        result = get_verified_vertex_models("my-project")

        assert result == {("anthropic", "claude-sonnet-4-5-20250514")}

    def test_get_verified_models_parses_raw_blob_bytes(self, monkeypatch):
        """Same as above but bypassing _mock_storage's dict->json.dumps step,
        so the exact bytes read off the blob are spelled out literally."""
        raw = (
            b'{"lamia-vertex-model-map": {"anthropic:claude-sonnet-4-5-20250514": "anthropic:claude-sonnet-4-5-20250514", '
            b'"google:gemini-2.5-flash": "google:gemini-2.5-flash"}}'
        )
        blob = MagicMock()
        blob.download_as_bytes.return_value = raw
        bucket = MagicMock()
        bucket.blob.return_value = blob
        client = MagicMock()
        client.bucket.return_value = bucket
        monkeypatch.setattr(
            "lamia_cloud.gcp.llm.vertex.storage.Client", lambda project: client
        )

        result = get_verified_vertex_models("my-project")

        assert result == {
            ("anthropic", "claude-sonnet-4-5-20250514"),
            ("google", "gemini-2.5-flash"),
        }

    def test_uses_dedicated_lamia_state_bucket(self, monkeypatch):
        client, _, _ = self._mock_storage(monkeypatch, {})

        get_verified_vertex_models("my-project")

        client.bucket.assert_called_once_with("my-project-lamia-state")

    def test_get_verified_models_missing_blob_returns_empty(self, monkeypatch):
        self._mock_storage(monkeypatch, existing=None)

        assert get_verified_vertex_models("my-project") == set()

    def test_get_verified_models_client_error_returns_empty(self, monkeypatch):
        def raise_error(project):
            raise RuntimeError("no credentials")

        monkeypatch.setattr("lamia_cloud.gcp.llm.vertex.storage.Client", raise_error)

        assert get_verified_vertex_models("my-project") == set()

    def test_remember_verified_models_merges_with_existing(self, monkeypatch):
        _, _, blob = self._mock_storage(
            monkeypatch, {"anthropic:claude-opus-4-20250514": "anthropic:claude-opus-4-20250514"}
        )

        remember_verified_vertex_models("my-project", {("anthropic", "claude-sonnet-4-5-20250514")})

        payload = json.loads(blob.upload_from_string.call_args[0][0])
        assert payload["lamia-vertex-model-map"] == {
            "anthropic:claude-opus-4-20250514": "anthropic:claude-opus-4-20250514",
            "anthropic:claude-sonnet-4-5-20250514": "anthropic:claude-sonnet-4-5-20250514",
        }

    def test_remember_verified_models_creates_bucket_if_missing(self, monkeypatch):
        client, bucket, _ = self._mock_storage(monkeypatch, existing=None)
        bucket.exists.return_value = False

        remember_verified_vertex_models("my-project", {("anthropic", "claude-sonnet-4-5-20250514")})

        client.create_bucket.assert_called_once()

    def test_remember_verified_models_retries_once_on_conflict(self, monkeypatch):
        _, _, blob = self._mock_storage(monkeypatch, {})
        blob.upload_from_string.side_effect = [PreconditionFailed("conflict"), None]

        remember_verified_vertex_models("my-project", {("anthropic", "claude-sonnet-4-5-20250514")})

        assert blob.upload_from_string.call_count == 2

    def test_remember_verified_models_empty_input_is_noop(self, monkeypatch):
        client, _, _ = self._mock_storage(monkeypatch, {})

        remember_verified_vertex_models("my-project", set())

        client.bucket.assert_not_called()

    def test_get_cached_resolution_returns_substitute(self, monkeypatch):
        self._mock_storage(monkeypatch, {
            "anthropic:claude-sonnet-4-5": "anthropic:claude-sonnet-4-5-20250514",
        })

        assert _get_cached_resolution("my-project", "anthropic", "claude-sonnet-4-5") == (
            "anthropic", "claude-sonnet-4-5-20250514",
        )

    def test_get_cached_resolution_none_when_not_cached(self, monkeypatch):
        self._mock_storage(monkeypatch, {})

        assert _get_cached_resolution("my-project", "anthropic", "claude-sonnet-4") is None

    def test_remember_resolved_models_records_substitution(self, monkeypatch):
        _, _, blob = self._mock_storage(monkeypatch, {})

        _remember_resolved_models("my-project", {
            ("anthropic", "claude-sonnet-4-5"): ("anthropic", "claude-sonnet-4-5-20250514"),
        })

        payload = json.loads(blob.upload_from_string.call_args[0][0])
        assert payload["lamia-vertex-model-map"] == {
            "anthropic:claude-sonnet-4-5": "anthropic:claude-sonnet-4-5-20250514",
        }

    def test_substitution_is_not_clobbered_by_later_identity_write(self, monkeypatch):
        """remember_verified_vertex_models (the flat CloudDeployer path) must
        not overwrite a real substitution _remember_resolved_models already
        recorded for the same key, even though both write to the same blob."""
        _, _, blob = self._mock_storage(
            monkeypatch, {"anthropic:claude-sonnet-4-5": "anthropic:claude-sonnet-4-5-20250514"}
        )

        remember_verified_vertex_models("my-project", {("anthropic", "claude-sonnet-4-5")})

        payload = json.loads(blob.upload_from_string.call_args[0][0])
        assert payload["lamia-vertex-model-map"]["anthropic:claude-sonnet-4-5"] == "anthropic:claude-sonnet-4-5-20250514"
