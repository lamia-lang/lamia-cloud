"""GCS bucket helpers."""
import logging

from google.cloud import storage

logger = logging.getLogger(__name__)


def ensure_bucket(project_id: str, bucket_name: str, location: str = "us") -> storage.Bucket:
    """Return a GCS bucket for this project, creating it if it doesn't exist."""
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(bucket_name, location=location)
        logger.info(f"Created bucket: {bucket_name}")
    return bucket
