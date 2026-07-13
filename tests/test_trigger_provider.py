"""Tests for lamia_cloud.gcp.trigger_provider — filter builders and failed-event handling.

These mock the GCP Pub/Sub and Monitoring clients rather than hitting real GCP:
the goal is verifying our own logic (filter construction, retry/ACK bookkeeping,
payload parsing), not the GCP SDK itself.
"""

import json
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import NotFound

from lamia_cloud.gcp.trigger_provider import (
    GCPTriggerProvider,
    _build_eventarc_filters,
    _build_pubsub_filter_expression,
)


class TestBuildEventarcFilters:
    def test_always_includes_type_filter(self):
        filters = _build_eventarc_filters(
            "file_created", {}, "google.cloud.storage.object.v1.finalized"
        )
        assert filters[0].attribute == "type"
        assert filters[0].value == "google.cloud.storage.object.v1.finalized"

    def test_adds_bucket_filter_for_storage_events_with_path(self):
        filters = _build_eventarc_filters(
            "file_created", {"path": "my-bucket"}, "google.cloud.storage.object.v1.finalized"
        )
        assert len(filters) == 2
        assert filters[1].attribute == "bucket"
        assert filters[1].value == "my-bucket"

    def test_no_bucket_filter_without_path(self):
        filters = _build_eventarc_filters(
            "file_created", {}, "google.cloud.storage.object.v1.finalized"
        )
        assert len(filters) == 1

    def test_no_storage_filters_for_non_storage_event(self):
        filters = _build_eventarc_filters(
            "email_received", {"path": "ignored"}, "google.cloud.pubsub.topic.v1.messagePublished"
        )
        assert len(filters) == 1


class TestBuildPubsubFilterExpression:
    def test_email_received_with_all_filters(self):
        expr = _build_pubsub_filter_expression("email_received", {
            "to": "pricing@company.com",
            "from_domain": "bigcorp.com",
            "subject_contains": "invoice",
            "label": "inbox",
        })
        assert 'attributes.recipient = "pricing@company.com"' in expr
        assert 'attributes.senderDomain = "bigcorp.com"' in expr
        assert 'attributes.subjectContains = "invoice"' in expr
        assert 'attributes.label = "inbox"' in expr
        assert expr.count(" AND ") == 3

    def test_email_received_no_filters_returns_empty_string(self):
        assert _build_pubsub_filter_expression("email_received", {}) == ""

    def test_non_email_method_ignores_config(self):
        assert _build_pubsub_filter_expression("file_created", {"to": "x@y.com"}) == ""


class TestGetFailedEvents:
    @patch("lamia_cloud.gcp.trigger_provider.pubsub_v1.SubscriberClient")
    def test_peeks_and_parses_json_payloads(self, mock_subscriber_cls):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber

        msg = MagicMock()
        msg.ack_id = "ack-1"
        msg.message.data = json.dumps({"sender": "a@b.com"}).encode("utf-8")
        msg.message.publish_time.isoformat.return_value = "2026-07-01T00:00:00Z"
        msg.message.attributes = {"retry_count": "3"}

        mock_response = MagicMock()
        mock_response.received_messages = [msg]
        mock_subscriber.pull.return_value = mock_response

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        events = provider.get_failed_events("my-trigger")

        assert events == [{
            "payload": {"sender": "a@b.com"},
            "timestamp": "2026-07-01T00:00:00Z",
            "attempt_count": 3,
        }]
        mock_subscriber.modify_ack_deadline.assert_called_once_with(
            subscription="projects/proj/subscriptions/lamia-trigger-my-trigger-dead-letter",
            ack_ids=["ack-1"],
            ack_deadline_seconds=0,
        )

    @patch("lamia_cloud.gcp.trigger_provider.pubsub_v1.SubscriberClient")
    def test_falls_back_to_raw_on_invalid_json(self, mock_subscriber_cls):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber

        msg = MagicMock()
        msg.ack_id = "ack-1"
        msg.message.data = b"not json"
        msg.message.publish_time.isoformat.return_value = "2026-07-01T00:00:00Z"
        msg.message.attributes = {}

        mock_response = MagicMock()
        mock_response.received_messages = [msg]
        mock_subscriber.pull.return_value = mock_response

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        events = provider.get_failed_events("my-trigger")

        assert events[0]["payload"] == {"raw": "not json"}

    @patch("lamia_cloud.gcp.trigger_provider.pubsub_v1.SubscriberClient")
    def test_returns_empty_list_when_no_messages(self, mock_subscriber_cls):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber
        mock_response = MagicMock()
        mock_response.received_messages = []
        mock_subscriber.pull.return_value = mock_response

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        assert provider.get_failed_events("my-trigger") == []
        mock_subscriber.modify_ack_deadline.assert_not_called()

    @patch("lamia_cloud.gcp.trigger_provider.pubsub_v1.SubscriberClient")
    def test_returns_empty_list_when_subscription_not_found(self, mock_subscriber_cls):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber
        mock_subscriber.pull.side_effect = NotFound("no sub")

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        assert provider.get_failed_events("my-trigger") == []


class TestClearFailedEvents:
    @patch("lamia_cloud.gcp.trigger_provider.pubsub_v1.SubscriberClient")
    def test_acknowledges_all_messages_across_pages(self, mock_subscriber_cls):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber

        page1 = MagicMock()
        page1.received_messages = [MagicMock(ack_id="a1"), MagicMock(ack_id="a2")]
        page2 = MagicMock()
        page2.received_messages = []
        mock_subscriber.pull.side_effect = [page1, page2]

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        count = provider.clear_failed_events("my-trigger")

        assert count == 2
        mock_subscriber.acknowledge.assert_called_once_with(
            subscription="projects/proj/subscriptions/lamia-trigger-my-trigger-dead-letter",
            ack_ids=["a1", "a2"],
        )

    @patch("lamia_cloud.gcp.trigger_provider.pubsub_v1.SubscriberClient")
    def test_returns_zero_when_subscription_not_found(self, mock_subscriber_cls):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber
        mock_subscriber.pull.side_effect = NotFound("no sub")

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        assert provider.clear_failed_events("my-trigger") == 0


class TestGetFailedEventCount:
    @patch("lamia_cloud.gcp.trigger_provider.monitoring_v3.MetricServiceClient")
    def test_returns_latest_point_value(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        ts = MagicMock()
        point = MagicMock()
        point.value.int64_value = 7
        ts.points = [point]
        mock_client.list_time_series.return_value = [ts]

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        assert provider._get_failed_event_count("my-trigger") == 7

    @patch("lamia_cloud.gcp.trigger_provider.monitoring_v3.MetricServiceClient")
    def test_returns_zero_when_no_time_series(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.list_time_series.return_value = []

        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        assert provider._get_failed_event_count("my-trigger") == 0

    @patch("lamia_cloud.gcp.trigger_provider.monitoring_v3.MetricServiceClient")
    def test_returns_zero_on_exception(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("boom")
        provider = GCPTriggerProvider(project_id="proj", location="us-central1")
        assert provider._get_failed_event_count("my-trigger") == 0
