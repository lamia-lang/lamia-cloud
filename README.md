# lamia-cloud

Cloud execution backend for [Lamia](https://github.com/lamia-lang/lamia). Run `.lm` scripts once with `--remote`, deploy scheduled cloud jobs, and prepare for upcoming cloud trigger support. Currently supports GCP.

For common agent use cases, you usually do not need to build custom cloud-agent infrastructure from scratch before shipping with Lamia.

## Installation

```bash
pip install "lamia-lang[cloud]"
```

## Prerequisites

- GCP project with billing enabled
- Application Default Credentials: `gcloud auth application-default login`

All required GCP APIs (including Service Usage) are enabled automatically on first deploy.

## Quick Start

1. Add a `cloud` section to your project's `config.yaml`:

```yaml
cloud:
  provider: gcp
  project_id: my-gcp-project
  location: us-central1  # optional, default: us-central1
```

2. Run a script once in the cloud with `--remote`:

```bash
lamia my_script.lm --remote
```

Use this one-shot run to validate cloud execution, permissions, and logs before adding a schedule.

3. Schedule your script with the `--remote` flag:

```bash
lamia schedule add my_script.lm --every day --remote
```

The `--remote` flag tells lamia to deploy and run the script in the cloud instead of locally.

## Managing Schedules

```bash
lamia schedule list              # shows all jobs (local + cloud) with live status
lamia schedule add X --remote    # deploy and schedule a new cloud job
lamia schedule remove <id>       # tears down cloud resources and removes the job
```

## How It Works

1. `lamia <script>.lm --remote` packages your project and runs it as a Cloud Run Job (one-shot)
2. `lamia schedule add <script>.lm --remote` deploys the same cloud job with Cloud Scheduler
3. Cloud Scheduler triggers the job on your cron schedule
4. Logs are available in Cloud Logging
5. `lamia schedule list` fetches live execution status from the cloud

## LLM on Cloud — Vertex AI

Scripts that use LLM calls run through **Vertex AI** on cloud. This gives you:

- **No API keys** — authentication via IAM, no keys to store, rotate, or leak
- **Budget control** — Vertex AI quotas and billing alerts
- **Secure by default** — no API key transport or storage, traffic stays within GCP

### Supported Models

| Provider | Cloud routing |
|----------|--------------|
| **Anthropic** (Claude) | Runs natively on Vertex AI — same models, same quality |
| **Google** (Gemini) | Runs natively on Vertex AI |
| **OpenAI** (GPT, o-series) | Automatically mapped to Gemini by tier (strong/medium/light) with runtime selection of the best available current Gemini model |

Anthropic and Google models run as-is. OpenAI models are mapped because they're not available on Vertex AI — tier classification is stable while the selected Gemini model is discovered dynamically at runtime.

## Configuration Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `cloud.provider` | Yes | — | Cloud provider (currently `gcp`) |
| `cloud.project_id` | Yes | — | Your GCP project ID |
| `cloud.location` | No | `us-central1` | Region for Cloud Run deployment |

No environment variables are required.

## Troubleshooting

- If Vertex AI access is not enabled yet, lamia-cloud logs a project-specific URL and attempts to open it automatically in your browser:
  `https://console.cloud.google.com/vertex-ai?project=<your-project-id>`
- After accepting terms, re-run the schedule/install command once.

## Development

```bash
git clone https://github.com/lamia-lang/lamia-cloud.git
cd lamia-cloud
pip install -e ".[dev]"
pytest
```

## Releasing

```bash
git tag v0.1.0
git push origin v0.1.0
```
