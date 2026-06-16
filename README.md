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
- Application Default Credentials: `gcloud auth application-default login`
- Service Usage API enabled (required to auto-enable everything else):
  ```bash
  gcloud services enable serviceusage.googleapis.com --project=YOUR_PROJECT_ID
  ```

All other required APIs (Cloud Build, Cloud Run, Cloud Scheduler, Vertex AI) are enabled automatically on first deploy.

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

## Configuration

The only configuration needed is in `config.yaml`:

| Field | Required | Description |
|-------|----------|-------------|
| `cloud.provider` | Yes | Cloud provider (`gcp`) |
| `cloud.project_id` | Yes | GCP project ID |
| `cloud.location` | No | GCP region for Cloud Run deployment (default: `us-central1`) |

Authentication uses Application Default Credentials (ADC).

### Region Handling

You only need to specify a single `location` — this is where your Cloud Run services are deployed. The LLM routing automatically handles provider-specific region requirements:

- **Google models (Gemini)**: uses your configured `location`
- **Anthropic models (Claude)**: automatically routes to `us-east5` (Anthropic's Vertex AI region), regardless of your Cloud Run location

This is transparent — your scripts don't need any changes.

### Vertex AI Model Access

To use LLM features on cloud, accept the Vertex AI Terms of Service:

1. Visit: https://console.cloud.google.com/vertex-ai?project=YOUR_PROJECT_ID
2. Accept the terms when prompted

lamia-cloud will attempt to open this URL automatically if model access fails.

### OpenAI Model Fallback

Scripts that use **OpenAI** models automatically fallback to Gemini modelswhen deployed to cloud. The reason for this is that OpenAI models are not available in Google Cloud as of now. Some of the mappings are:

| OpenAI Model | Vertex AI Equivalent |
|-------------|---------------------|
| gpt-4o, gpt-4o-mini | gemini-2.0-flash |
| gpt-4, o1-preview | gemini-1.5-pro |
| gpt-3.5-turbo | gemini-2.0-flash |

Anthropic models (Claude) route to Vertex AI's Anthropic endpoint in `us-east5`.

## Security

Cloud Run services are deployed with:
- **Internal-only ingress** — only accepts traffic from within GCP (Cloud Scheduler, VPC)
- **IAM authentication** — Cloud Scheduler authenticates via OIDC token signed by `lamia-runner`
- **No public access** — services are never exposed to the internet
- **No secrets in images** — `.env` files are excluded from container builds; API keys are managed via IAM

### Service Account

lamia-cloud creates a dedicated `lamia-runner` service account with minimal permissions:
- `roles/aiplatform.user` — Vertex AI model access
- `roles/run.invoker` — allows scheduler to invoke Cloud Run services

## What Happens

- Your `.lm` script is packaged and deployed as a Cloud Run service with the lamia runtime
- Cloud Scheduler triggers it on your cron schedule via authenticated OIDC request
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
