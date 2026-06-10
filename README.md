# lamia-cloud

Cloud scheduling backend for [Lamia](https://github.com/lamia-lang/lamia). Scripts run entirely in the cloud — deployed and triggered automatically. Currently supports GCP.

## Installation

```bash
pip install "lamia-lang[cloud]"
```

Or install this package directly:

```bash
pip install lamia-cloud
```

## Prerequisites

- A GCP project with **billing enabled**
  ```bash
  gcloud billing projects link YOUR_PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
  ```
  To find your billing account ID: `gcloud billing accounts list`
- Application Default Credentials: `gcloud auth application-default login`

Enable the Service Usage API once (required to auto-enable everything else):

```bash
gcloud services enable serviceusage.googleapis.com --project=YOUR_PROJECT_ID
```

All other required APIs (Cloud Build, Cloud Run, Cloud Scheduler, etc.) are enabled automatically on first deploy.

## Quick Start

1. Add a `cloud` section to your project's `config.yaml`:

```yaml
cloud:
  provider: gcp
  project_id: my-gcp-project
  location: us-central1
```

2. Authenticate:

```bash
gcloud auth application-default login
```

3. Schedule (deploys + creates trigger automatically):

```bash
lamia schedule add my_script.lm --every day --remote
```

## What Happens

- Your `.lm` script is packaged and deployed as a Cloud Run service with the lamia runtime
- Cloud Scheduler triggers it on your cron schedule
- Logs flow to Cloud Logging
- `lamia schedule list` shows status alongside local jobs
- `lamia schedule remove <id>` tears everything down

## Development

```bash
git clone https://github.com/lamia-lang/lamia-cloud.git
cd lamia-cloud
pip install -e ".[dev]"
pytest
```

## Releasing

Tag a version and push — CI handles the rest:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow builds and publishes to PyPI automatically using trusted publishing.
