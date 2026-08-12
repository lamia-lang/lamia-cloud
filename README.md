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

## CI Authentication Architecture

When lamia detects a CI environment (GitHub Actions), it authenticates to the cloud provider without static credentials. This section describes the GCP implementation; other providers follow the same pattern through the `CloudDeployer` interface.

### Trust Model

```
GitHub Actions                  Lamia                     GCP
     |                            |                        |
     |  OIDC token (short-lived)  |                        |
     |--------------------------->|                        |
     |                            |  STS token exchange    |
     |                            |----------------------->|
     |                            |  GCP access token      |
     |                            |<-----------------------|
     |                            |  deploy/run            |
     |                            |----------------------->|
```

GitHub's OIDC token is cryptographically signed and contains claims about the repository, branch, and workflow. GCP's Workload Identity Federation validates these claims against a pre-configured trust policy. No long-lived credentials exist anywhere in this flow.

### What `lamia cloud connect` Creates (GCP)

| Resource | Purpose | Naming Convention |
|----------|---------|-------------------|
| WIF Pool | Shared identity pool for all lamia-connected repos in the project | `lamia-github-pool` |
| WIF Provider | Per-repo OIDC trust with branch restriction | `lamia-gh-{sanitized-repo}` |
| CI Service Account | Deploy permissions: build containers, create/update Cloud Run Jobs | `lm-ci-{sanitized-repo}` |
| Runtime Service Account | Minimal permissions: only what deployed scripts need (e.g. Vertex AI) | `lm-run-{sanitized-repo}` |
| Cloud Build Connection | GitHub App installation for source cloning | `lamia-github` |

### Security Decisions

**Per-repo service accounts (not shared).** Each connected repository gets its own CI and runtime service accounts. If repository A is compromised, its service account cannot access resources deployed by repository B in the same GCP project. This prevents lateral movement between repositories.

**Deploy/runtime SA separation.** The CI service account (`lm-ci-*`) has permissions to build containers and deploy Cloud Run Jobs. The runtime service account (`lm-run-*`) has only the permissions the script needs (e.g. `roles/aiplatform.user` for Vertex AI). Deployed code cannot redeploy, modify infrastructure, or escalate its own permissions.

**Branch-scoped WIF condition.** The WIF provider's attribute condition restricts authentication to a specific repository AND branch:
```
assertion.repository == "owner/repo" && assertion.ref == "refs/heads/main"
```
Feature branches, forks, and pull requests cannot obtain credentials even if a workflow runs.

**Subprocess error checking.** All `gcloud` subprocess calls check exit codes and raise on failure. Silent failures during WIF setup are not possible.

**Runtime validation.** At CI time, lamia validates the workspace git remote against the connected repository stored in `config.yaml`. This catches git remote tampering where an attacker changes `origin` to point to a different repository while using the victim's WIF credentials.

**`pull_request_target` rejection.** Lamia explicitly refuses to authenticate when the GitHub event is `pull_request_target`. This event runs in the base repository context and would allow a fork PR to deploy with the base repo's credentials.

**Credential file hygiene.** Temporary credential files are created with `0600` permissions (owner-only read/write) and cleaned up via `atexit` handler when the process exits.

### Open-Source vs Private Repositories

**Public repositories** face the highest risk because anyone can submit a pull request. The branch restriction in WIF is critical — only code merged to `main` can trigger deployments. Maintainers must enforce code reviews and branch protection rules. The WIF trust model prevents fork PRs from authenticating, but a malicious contribution merged to main by a compromised maintainer account would deploy.

**Private repositories** benefit from access control limiting who can push or merge. The primary risks are compromised collaborator accounts and credential leaks. Per-repo SA isolation limits blast radius. For private repos, the branch restriction may be relaxed (e.g. allowing `develop` branch deploys) since contributors are trusted.

### Disconnect and Revocation

`lamia cloud disconnect` removes the WIF provider, both service accounts, and the Cloud Build repository link. The shared WIF pool is left intact (other repos may use it). Connection details are removed from `config.yaml`.

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
