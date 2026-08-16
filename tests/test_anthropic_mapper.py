"""Tests for Anthropic model id parsing, catalog display naming, and version selection."""

from lamia_cloud.gcp.llm.anthropic_mapper import model_garden_name, select_model


class TestModelGardenName:
    def test_current_scheme_with_minor(self):
        assert model_garden_name("claude-sonnet-4-5-20250929") == "Claude Sonnet 4.5"

    def test_current_scheme_without_minor(self):
        assert model_garden_name("claude-sonnet-4-20250514") == "Claude Sonnet 4"

    def test_current_scheme_no_date_no_minor(self):
        assert model_garden_name("claude-sonnet-5") == "Claude Sonnet 5"

    def test_current_scheme_no_date_with_minor(self):
        assert model_garden_name("claude-opus-4-1") == "Claude Opus 4.1"

    def test_legacy_scheme_with_minor(self):
        assert model_garden_name("claude-3-5-haiku-20241022") == "Claude 3.5 Haiku"

    def test_legacy_scheme_without_minor(self):
        assert model_garden_name("claude-3-opus-20240229") == "Claude 3 Opus"

    def test_unparseable_id_falls_back_unchanged(self):
        assert model_garden_name("some-future-shape-v9") == "some-future-shape-v9"


class TestSelectModel:
    def test_exact_match_returned_unchanged(self):
        available = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]
        assert select_model("claude-sonnet-4-5-20250929", available) == "claude-sonnet-4-5-20250929"

    def test_no_catalog_returns_unchanged(self):
        assert select_model("claude-sonnet-4-5-20250929", []) == "claude-sonnet-4-5-20250929"

    def test_nearest_version_same_generation_preferred_over_next_generation(self):
        # sonnet-4-6 is one minor version away; sonnet-5 is a whole
        # generation away -- nearest by distance should win.
        available = ["claude-sonnet-5", "claude-sonnet-4-6"]
        assert select_model("claude-sonnet-4-5-20250929", available) == "claude-sonnet-4-6"

    def test_nearest_version_picks_closest_minor(self):
        available = ["claude-opus-4-7", "claude-opus-4-8", "claude-opus-5"]
        assert select_model("claude-opus-4-6", available) == "claude-opus-4-7"

    def test_cross_era_upgrade_legacy_to_current(self):
        # Legacy 3.5 Haiku retired; only the current-scheme 4.5 exists.
        available = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]
        assert select_model("claude-3-5-haiku-20241022", available) == "claude-haiku-4-5-20251001"

    def test_family_substring_fallback_when_no_family_member_parses(self):
        # Catalog entry has an unexpected suffix that doesn't match either
        # id-scheme regex, but still contains the family name.
        available = ["claude-opus-5-preview-exp"]
        assert select_model("claude-opus-4-6", available) == "claude-opus-5-preview-exp"

    def test_family_substring_fallback_for_unparseable_request(self):
        available = ["claude-haiku-4-5-20251001", "claude-haiku-latest", "claude-sonnet-5"]
        assert select_model("claude-haiku-latest-preview", available) == "claude-haiku-4-5-20251001"

    def test_family_substring_fallback_picks_highest_version(self):
        available = ["claude-opus-4-7", "claude-opus-4-8"]
        assert select_model("claude-opus-unparseable-id", available) == "claude-opus-4-8"

    def test_no_family_match_returns_unchanged(self):
        available = ["claude-sonnet-5"]
        assert select_model("claude-opus-4-6", available) == "claude-opus-4-6"

    def test_completely_unrelated_request_returns_unchanged(self):
        available = ["claude-sonnet-5", "claude-opus-5"]
        assert select_model("not-a-claude-model", available) == "not-a-claude-model"
