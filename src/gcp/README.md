# GCP Backend — Architecture & Technical Reference

This document describes the internal architecture of the GCP cloud backend. It is
maintained for developers and AI systems that need to understand how lamia-cloud
orchestrates cloud execution. **User-facing documentation lives in `lamia/docs/`.**

---

## Overview of Execution Modes

| Mode | Entry point | GCP services |
|------|-------------|--------------|
| One-shot (`--remote`) | `deployer.py` | Cloud Build → Cloud Run Job |
| Scheduled (`schedule add --remote`) | `scheduler.py` + `deployer.py` | Cloud Scheduler → Cloud Run Job |
| Trigger (reactive) | `trigger_provider.py` | Eventarc → Workflows → Cloud Run Job |
| Trigger (scheduled / employee mode) | `trigger_provider.py` | Eventarc → Pub/Sub → Cloud Scheduler → Workflows → Cloud Run Job |

---

## Triggers — Full Technical Design

### Concepts

A **trigger** is a `trigger.*` call inside a `.lm` script. A script can contain one or
more triggers placed anywhere in the code. Each trigger boundary splits the script into
**stages** — each stage becomes a separate Cloud Run Job container.

```
trigger.email_received(sender, subject, body, to="pricing@co.com")
# ─── stage 0 code ───
... actions ...
trigger.file_created(name, size, path="sales/custom-pricing")
# ─── stage 1 code ───
... more actions ...
```

### Deployment Modes

#### Reactive (always-on)

Deployed via `lamia script.lm --remote` when the script starts with a trigger.

```
Eventarc ──event──▶ Workflow ──▶ Cloud Run Job (stage 0)
                       │
                       ├── wait on per-execution Pub/Sub subscription
                       │
                       └──▶ Cloud Run Job (stage 1) ──▶ ...
```

- Eventarc routes the matching event directly to a Workflow execution.
- Single-stage scripts: workflow runs one Job and returns.
- Multi-stage scripts: workflow creates per-execution subscriptions and waits.

#### Scheduled (employee mode)

Deployed via `lamia schedule add script.lm --every day --remote` when the script
starts with a trigger.

```
Eventarc ──event──▶ Pub/Sub accumulation topic
                              │
Cloud Scheduler ──cron──▶ Drain Workflow ──pulls──▶ [messages]
                              │
                              └── parallel: Cloud Run Job per message
```

- Events accumulate in a Pub/Sub subscription between scheduler activations.
- At cron time, the drain workflow pulls all pending messages in batches of 100.
- Each message fans out to a parallel Cloud Run Job execution.
- Mimics human working patterns (wake → process backlog → sleep).

---

### Resource Naming Convention

All GCP resources use the prefix `lamia-trigger-{plan_name}`:

| Resource | Name pattern |
|----------|--------------|
| Workflow | `lamia-trigger-{name}` |
| Eventarc trigger | `lamia-trigger-{name}` |
| Cloud Run Job (single stage) | `lamia-{name}` |
| Cloud Run Job (multi-stage) | `lamia-{name}-stage-{i}` |
| Accumulation topic (scheduled) | `lamia-trigger-{name}-events` |
| Accumulation subscription | `lamia-trigger-{name}-events` |
| Continuation topic (stage N) | `lamia-trigger-{name}-stage-{N}` |
| Per-execution subscription | `lamia-trigger-{name}-stage-{N}-{exec_id}` |
| Dead-letter topic | `lamia-trigger-{name}-dead-letter` |
| Dead-letter subscription | `lamia-trigger-{name}-dead-letter` |
| Cloud Scheduler job | `lamia-trigger-{name}-scheduler` |
| Service account | `lamia-runner@{project}.iam.gserviceaccount.com` |

---

### Multi-Stage Isolation (Per-Execution Subscriptions)

When a script has continuation triggers (mid-script `trigger.*` calls), multiple
workflow executions may wait for events concurrently. To prevent message stealing:

1. At deploy time, only **topics** are created for continuation stages.
2. At runtime, each workflow execution derives a unique `exec_id` from
   `GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID`.
3. The workflow dynamically creates a subscription named
   `{topic}-{exec_id}` on the continuation topic.
4. Only this execution pulls from that subscription.
5. After the event is processed, the subscription is deleted.
6. Safety net: subscriptions have `expirationPolicy.ttl` of 72h + 1h buffer.

This guarantees that 10 concurrent executions waiting for "a file in bucket X" each
get only their own matching file event, with zero interference.

---

### ACK / NACK Protocol

Messages are **never** acknowledged before the job finishes. The outcome determines
the signal sent back to Pub/Sub:

| Job exit code | Meaning | Workflow action |
|---------------|---------|-----------------|
| 0 | Success | ACK immediately → proceed to next stage |
| 2 | Reject (`trigger.reject()`) | ACK immediately → loop back to pull next message |
| 1 (or any other) | Unhandled exception | Retry up to N times → then ACK + dead-letter |

#### Exit code mapping (Lamia runtime)

- `TriggerRejectError` (raised by `trigger.reject()`) → process exits with code **2**
- Normal completion → process exits with code **0**
- Unhandled exception → process exits with code **1**

This is handled in `lamia/cli/cli.py` — the `TriggerRejectError` handler calls
`_graceful_shutdown(lamia, 2)`.

#### Single-stage reactive workflows

For single-stage triggers, the Eventarc event is not a Pub/Sub message (it's a direct
invocation). The workflow still distinguishes exit codes:

- Exit 0: workflow returns normally.
- Exit 2: workflow returns `"Event rejected by script"` (no error propagated).
- Exit 1: workflow re-raises the error (visible in Workflow execution logs).

---

### Retry Logic and Failed Events

**Constants** (defined in `workflow_generator.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_EXCEPTION_RETRIES` | 5 | Max times a failing job is retried for the same message |
| `EXIT_CODE_REJECT` | 2 | Exit code signaling `trigger.reject()` |
| `CONTINUATION_TIMEOUT_SECONDS` | 259200 (72h) | Max wait time for continuation events |

**Retry flow for continuation stages:**

```
pull message
  → run job
    → exit 0 → ACK → proceed
    → exit 2 → ACK → pull next (reject loop)
    → exit 1 → retry_count++
                → if retry_count >= 5:
                    ACK + publish to dead-letter topic → raise error
                → else: loop back to pull (same message redelivered after ack deadline)
```

**User-facing abstraction: "failed events"**

Users never see "dead letter" terminology. The interface exposes:

| CLI command | What it does |
|-------------|--------------|
| `lamia trigger list` | Shows count of failed events per trigger |
| `lamia trigger list --verbose` | Shows count + actual event payloads + timestamps |
| `lamia trigger clear-failed <name>` | Acknowledges and removes all failed events |

**Provider interface** (`CloudTriggerProvider`):

| Method | Return | Description |
|--------|--------|-------------|
| `get_failed_events(name)` | `List[dict]` | Peek event payloads without consuming (returns payload + timestamp) |
| `clear_failed_events(name)` | `int` | Ack all messages, return count removed |

**GCP implementation detail** (internal — users don't see this):

- Dead-letter Pub/Sub topic: `lamia-trigger-{name}-dead-letter`
- Created at deploy time with a paired subscription.
- Failed events are published here after retries are exhausted.
- Count is queried via Stackdriver Monitoring (`num_undelivered_messages` metric).
- `get_failed_events()` uses `subscriber.pull()` + `modify_ack_deadline(0)` to peek
  without consuming. Messages remain in the subscription.
- `clear_failed_events()` pulls and acks in batches of 100 until empty.
- Messages persist until manually cleared (no automatic expiration configured).
- Other providers (AWS, Azure) would implement the same interface using their own
  dead-letter mechanisms (SQS DLQ, Service Bus dead-letter queue, etc.).

---

### Drain Workflow (Scheduled Mode) — ACK Behavior

In the drain workflow, each message in a pulled batch is processed in parallel:

```
for each message in batch (parallel):
  decode → run job
    → success: ACK the individual message
    → failure: ACK + publish to dead-letter (no infinite redelivery loops)
```

After all parallel jobs complete, the workflow loops to pull the next batch.
When no messages remain, the workflow exits with a processed count.

---

### Infrastructure-Level Filters

Filters prevent the script from launching at all for non-matching events.
They are applied at two layers:

**1. Eventarc filters (for storage events):**

| Trigger config key | Eventarc attribute |
|--------------------|-------------------|
| `path` | `bucket` |

Example: `trigger.file_created(name, size, path="my-bucket")` sets
`EventFilter(attribute="bucket", value="my-bucket")`.

**2. Pub/Sub subscription filters (for email/Pub/Sub events):**

| Trigger config key | Pub/Sub attribute |
|--------------------|-------------------|
| `to` | `attributes.recipient` |
| `from_domain` | `attributes.senderDomain` |
| `subject_contains` | `attributes.subjectContains` |
| `label` | `attributes.label` |

Multiple filters are AND-combined. Example:
`trigger.email_received(sender, body, to="sales@co.com", from_domain="bigcorp.com")`
produces filter: `attributes.recipient = "sales@co.com" AND attributes.senderDomain = "bigcorp.com"`.

---

### Event Data Passing

Events are passed to Cloud Run Jobs via environment variable:

| Env var | Contents | Set by |
|---------|----------|--------|
| `LAMIA_TRIGGER_EVENT` | JSON-encoded event payload | Workflow container override |
| `LAMIA_STAGE_CONTEXT` | JSON-encoded previous stage result | Workflow (stages > 0) |
| `LAMIA_SCRIPT` | Script filename (e.g. `my_script.lm`) | Dockerfile CMD |

The Lamia runtime reads `LAMIA_TRIGGER_EVENT` in `TriggerActions._resolve()` and
injects fields as local variables via AST transformation.

---

### Timeout Behavior

| Scenario | Timeout | What happens |
|----------|---------|--------------|
| Continuation wait (pull loop) | 72 hours | Workflow polls every 30s. After 72h the subscription expires. |
| Per-execution subscription TTL | 73 hours | Safety expiration if workflow crashes without cleanup. |
| Ack deadline (per message) | 600 seconds | If job doesn't finish in 10 min, message becomes eligible for redelivery. |
| Cloud Run Job max execution | Configured per job | Depends on script requirements (default: 10 min, max: 24h). |

---

### Continuation Pull Loop (Workflow Steps)

For each continuation stage N, the generated workflow contains:

```yaml
# Abbreviated — see workflow_generator.py for full output
pull_stage_N:        # Pull one message from per-execution subscription
check_message_N:     # If empty → sleep 30s → goto pull_stage_N
decode_event_N:      # Extract event data + store ack_id
run_stage_N:         # try: run Cloud Run Job
                     #   except (exit 2): ACK → goto pull_stage_N (reject)
                     #   except (other):  retry_count++ → dead-letter or retry
ack_success_N:       # ACK on successful completion
cleanup_sub_stage_N: # Delete ephemeral subscription
```

---

### Event Method Mapping

| `trigger.*` method | Eventarc event type |
|--------------------|---------------------|
| `file_created` | `google.cloud.storage.object.v1.finalized` |
| `file_deleted` | `google.cloud.storage.object.v1.deleted` |
| `file_modified` | `google.cloud.storage.object.v1.metadataUpdated` |
| `email_received` | `google.cloud.pubsub.topic.v1.messagePublished` |

---

### Undeploy — Resources Removed

`GCPTriggerProvider.undeploy(name)` removes:

1. Workflow (`lamia-trigger-{name}`)
2. Eventarc trigger (`lamia-trigger-{name}`)
3. Cloud Scheduler job (if scheduled mode)
4. Accumulation Pub/Sub topic + subscription (if scheduled mode)
5. Dead-letter Pub/Sub topic + subscription
6. All Cloud Run Jobs (via `deployer.teardown`)

Per-execution subscriptions are ephemeral and self-expire (TTL), but are also
explicitly deleted by the workflow on normal completion.

---

## One-Shot Execution (`--remote`)

Simple flow, no triggers involved:

1. `deployer.py` packages the project → Cloud Build → Cloud Run Job
2. `run_job()` invokes the job and polls until completion
3. `fetch_execution_logs()` retrieves stdout/stderr from Cloud Logging
4. Exit code is propagated to the local CLI

Incremental deploys: source hash is stored as a label on the job. If unchanged,
the build step is skipped.

---

## Scheduled Execution

1. `scheduler.py` wraps `deployer.py` deployment + Cloud Scheduler creation.
2. Cloud Scheduler calls `jobs.run` on the Cloud Run Job at cron time.
3. `lamia schedule list` with `--remote` queries Scheduler + Job state.

---

## LLM Routing (Vertex AI)

- Scripts calling LLM use Vertex AI on cloud (no API keys needed).
- Authentication: IAM-based via service account.
- Anthropic/Google models run natively on Vertex AI.
- OpenAI models are mapped to equivalent-tier Gemini models.
- See `vertex.py` for the dynamic model mapping logic.

---

## File Structure

```
src/gcp/
├── README.md               ← this file
├── __init__.py             ← exports GCPCloudScheduler, GCPTriggerProvider, VertexLLM
├── deployer.py             ← Cloud Build + Cloud Run Job deployment
├── scheduler.py            ← Cloud Scheduler integration
├── trigger_provider.py     ← Trigger orchestration (Eventarc, Workflows, Pub/Sub)
├── workflow_generator.py   ← YAML generation for Workflows (reactive + drain)
└── vertex.py               ← Vertex AI LLM routing
```

---

## Type Schemas

### TriggerDeploymentPlan

```python
@dataclass
class TriggerDeploymentPlan:
    name: str                              # slug derived from script filename
    stages: List[TriggerStage]             # ordered list of trigger boundaries
    capabilities: dict                     # {uses_llm, uses_browser, uses_files, ...}
    mode: str = "reactive"                 # "reactive" or "scheduled"
    cron: Optional[str] = None             # cron expression (scheduled mode only)
```

### TriggerStage

```python
@dataclass
class TriggerStage:
    stage_index: int                       # 0-based position in script
    trigger_method: str                    # e.g. "email_received", "file_created"
    trigger_config: dict                   # string kwargs → infra filters
    output_bindings: List[str]             # variable names injected by AST transform
    script_source: str = ""                # source code for this stage
```

---

## Modifying This Architecture

When changing trigger behavior:

1. Update `workflow_generator.py` for orchestration logic changes.
2. Update `trigger_provider.py` for resource provisioning changes.
3. Update `types.py` if the deployment plan schema changes.
4. Update `lamia/actions/trigger.py` if runtime interface changes.
5. Update `lamia/docs/user-guide/triggers.md` for user-facing changes.
6. **Update this README** to keep the technical reference in sync.
