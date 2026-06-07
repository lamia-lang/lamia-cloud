# lamia-cloud

Cloud scheduling backend for [Lamia](https://github.com/lamia-lang/lamia). Currently supports GCP Cloud Scheduler.

## Installation

```bash
pip install "lamia-lang[cloud]"
```

Or install this package directly:

```bash
pip install lamia-cloud
```

## Quick Start

1. Add a `cloud` section to your project's `config.yaml`:

```yaml
cloud:
  provider: gcp
  project_id: my-gcp-project
  location: us-central1
  target_url: https://my-cloud-run-service.run.app/schedule
```

2. Authenticate:

```bash
gcloud auth application-default login
```

3. Schedule:

```bash
lamia schedule add my_script.lm --every day --remote
```

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

The GitHub Actions workflow builds the package and publishes to PyPI automatically using trusted publishing (no API tokens needed — configure trusted publishing in your PyPI project settings).

### Keeping lamia up-to-date with the latest lamia-cloud

For development, install from the local checkout:

```bash
pip install -e /path/to/lamia-cloud
```

For production, pin to a version range in your requirements:

```
lamia-cloud>=0.1.0
```

Or always get the latest:

```bash
pip install --upgrade lamia-cloud
```
