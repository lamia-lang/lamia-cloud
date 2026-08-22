"""Tests for shared GCS bucket helpers."""
from unittest.mock import MagicMock, patch

from lamia_cloud.gcp.storage_utils import ensure_bucket


class TestEnsureBucket:
    @patch("lamia_cloud.gcp.storage_utils.storage.Client")
    def test_returns_existing_bucket_without_creating(self, mock_client_cls):
        bucket = MagicMock()
        bucket.exists.return_value = True
        client = MagicMock()
        client.bucket.return_value = bucket
        mock_client_cls.return_value = client

        result = ensure_bucket("my-project", "my-bucket")

        assert result is bucket
        client.create_bucket.assert_not_called()

    @patch("lamia_cloud.gcp.storage_utils.storage.Client")
    def test_creates_bucket_when_missing(self, mock_client_cls):
        existing_bucket = MagicMock()
        existing_bucket.exists.return_value = False
        created_bucket = MagicMock()
        client = MagicMock()
        client.bucket.return_value = existing_bucket
        client.create_bucket.return_value = created_bucket
        mock_client_cls.return_value = client

        result = ensure_bucket("my-project", "my-bucket", location="us")

        client.create_bucket.assert_called_once_with("my-bucket", location="us")
        assert result is created_bucket

    @patch("lamia_cloud.gcp.storage_utils.storage.Client")
    def test_default_location_is_us(self, mock_client_cls):
        bucket = MagicMock()
        bucket.exists.return_value = False
        client = MagicMock()
        client.bucket.return_value = bucket
        mock_client_cls.return_value = client

        ensure_bucket("my-project", "my-bucket")

        client.create_bucket.assert_called_once_with("my-bucket", location="us")
