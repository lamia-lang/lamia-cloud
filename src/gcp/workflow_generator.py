"""Generate Google Cloud Workflow YAML from a TriggerDeploymentPlan.

Single-trigger reactive: Eventarc starts workflow, workflow runs Job with env override.
Multi-trigger reactive: After each Job, workflow waits via isolated per-execution
    Pub/Sub subscription for the next continuation event.
Drain (scheduled): Pulls all accumulated events from Pub/Sub, fans out one Job per event.
"""

import yaml

from lamia_cloud.types import TriggerDeploymentPlan

CONTINUATION_TIMEOUT_SECONDS = 259200  # 72 hours
MAX_EXCEPTION_RETRIES = 5
EXIT_CODE_REJECT = 2


def generate_workflow_yaml(
    plan: TriggerDeploymentPlan,
    project_id: str,
    location: str,
) -> str:
    """Generate reactive Workflow definition YAML for the given trigger plan."""
    if len(plan.stages) == 1:
        return _single_stage_workflow(plan, project_id, location)
    return _multi_stage_workflow(plan, project_id, location)


def generate_drain_workflow_yaml(
    plan: TriggerDeploymentPlan,
    project_id: str,
    location: str,
) -> str:
    """Generate drain Workflow YAML for scheduled mode (employee mode).

    The workflow:
    1. Pulls all pending messages from the accumulation subscription
    2. For each message, launches a Cloud Run Job (fan-out)
    3. Each Job may have its own continuation triggers (mid-script waits)
    4. Repeats until queue is empty
    """
    total_stages = len(plan.stages)
    subscription = f"projects/{project_id}/subscriptions/lamia-trigger-{plan.name}-events"
    job_name = _job_resource_name(project_id, location, plan.name, 0, total_stages)

    steps = []

    steps.append({
        "init": {
            "assign": [
                {"total_processed": 0},
                {"keep_draining": True},
            ],
        },
    })

    steps.append({
        "drain_loop": {
            "steps": [
                {
                    "pull_batch": {
                        "call": "googleapis.pubsub.v1.projects.subscriptions.pull",
                        "args": {
                            "subscription": subscription,
                            "body": {"maxMessages": 100},
                        },
                        "result": "pull_response",
                    },
                },
                {
                    "check_empty": {
                        "switch": [{
                            "condition": "${not(\"receivedMessages\" in pull_response.body) or len(pull_response.body.receivedMessages) == 0}",
                            "steps": [{
                                "mark_done": {
                                    "assign": [{"keep_draining": False}],
                                },
                            }],
                        }],
                    },
                },
                {
                    "process_batch": {
                        "switch": [{
                            "condition": "${keep_draining}",
                            "steps": _build_fan_out_steps(
                                job_name, plan, project_id, location, subscription, total_stages,
                            ),
                        }],
                    },
                },
                {
                    "continue_or_exit": {
                        "switch": [{
                            "condition": "${keep_draining}",
                            "next": "drain_loop",
                        }],
                    },
                },
            ],
        },
    })

    steps.append({
        "return_summary": {
            "return": "${\"Processed \" + string(total_processed) + \" events\"}",
        },
    })

    workflow = {
        "main": {
            "params": ["event"],
            "steps": steps,
        },
    }
    return yaml.dump(workflow, default_flow_style=False, sort_keys=False)


def _build_fan_out_steps(
    job_name: str,
    plan: TriggerDeploymentPlan,
    project_id: str,
    location: str,
    subscription: str,
    total_stages: int,
) -> list:
    """Build workflow steps that process each message in the pulled batch.

    Each message is acked individually after its job completes successfully.
    Failed jobs cause the message to be nacked (not acked) so Pub/Sub redelivers.
    """
    dead_letter_topic = f"projects/{project_id}/topics/lamia-trigger-{plan.name}-dead-letter"

    steps = []
    steps.append({
        "fan_out": {
            "parallel": {
                "for": {
                    "value": "msg",
                    "in": "${pull_response.body.receivedMessages}",
                    "steps": [
                        {
                            "decode_event": {
                                "assign": [{
                                    "event_data": "${json.decode(base64.decode(msg.message.data))}",
                                }],
                            },
                        },
                        {
                            "run_job": {
                                "try": {
                                    "steps": [{
                                        "execute": {
                                            "call": "googleapis.run.v2.projects.locations.jobs.run",
                                            "args": {
                                                "name": job_name,
                                                "body": {
                                                    "overrides": {
                                                        "containerOverrides": [{
                                                            "env": [{
                                                                "name": "LAMIA_TRIGGER_EVENT",
                                                                "value": "${json.encode(event_data)}",
                                                            }],
                                                        }],
                                                    },
                                                },
                                            },
                                            "result": "job_result",
                                        },
                                    }],
                                    "except": {
                                        "as": "job_error",
                                        "steps": [{
                                            "publish_failed": {
                                                "call": "googleapis.pubsub.v1.projects.topics.publish",
                                                "args": {
                                                    "topic": dead_letter_topic,
                                                    "body": {
                                                        "messages": [{"data": "${base64.encode(json.encode(event_data))}"}],
                                                    },
                                                },
                                            },
                                        }],
                                    },
                                },
                            },
                        },
                        {
                            "ack_message": {
                                "call": "googleapis.pubsub.v1.projects.subscriptions.acknowledge",
                                "args": {
                                    "subscription": subscription,
                                    "body": {"ackIds": ["${msg.ackId}"]},
                                },
                            },
                        },
                    ],
                },
            },
        },
    })

    steps.append({
        "update_count": {
            "assign": [{
                "total_processed": "${total_processed + len(pull_response.body.receivedMessages)}",
            }],
        },
    })

    return steps


def _job_resource_name(project_id: str, location: str, plan_name: str, stage_index: int, total_stages: int) -> str:
    if total_stages == 1:
        return f"projects/{project_id}/locations/{location}/jobs/lamia-{plan_name}"
    return f"projects/{project_id}/locations/{location}/jobs/lamia-{plan_name}-stage-{stage_index}"


def _single_stage_workflow(plan: TriggerDeploymentPlan, project_id: str, location: str) -> str:
    """Single-stage: Eventarc triggers workflow, workflow runs one Job.

    On success (exit 0): workflow returns normally.
    On reject (exit 2): workflow returns quietly (event doesn't match filter).
    On exception (exit 1): workflow propagates the error for alerting/monitoring.
    """
    job_name = _job_resource_name(project_id, location, plan.name, 0, 1)
    workflow = {
        "main": {
            "params": ["event"],
            "steps": [
                {
                    "run_stage": {
                        "try": {
                            "steps": [{
                                "execute_job": {
                                    "call": "googleapis.run.v2.projects.locations.jobs.run",
                                    "args": {
                                        "name": job_name,
                                        "body": {
                                            "overrides": {
                                                "containerOverrides": [{
                                                    "env": [{
                                                        "name": "LAMIA_TRIGGER_EVENT",
                                                        "value": "${json.encode(event)}",
                                                    }],
                                                }],
                                            },
                                        },
                                    },
                                    "result": "execution",
                                },
                            }],
                            "except": {
                                "as": "e",
                                "steps": [{
                                    "check_reject": {
                                        "switch": [{
                                            "condition": f'${{"exitCode" in e and e.exitCode == {EXIT_CODE_REJECT}}}',
                                            "steps": [{
                                                "return_rejected": {
                                                    "return": "${\"Event rejected by script\"}",
                                                },
                                            }],
                                        }],
                                    },
                                }, {
                                    "propagate_error": {
                                        "raise": "${e}",
                                    },
                                }],
                            },
                        },
                    },
                },
                {
                    "return_result": {
                        "return": "${execution}",
                    },
                },
            ],
        },
    }
    return yaml.dump(workflow, default_flow_style=False, sort_keys=False)


def _build_continuation_loop(
    stage_index: int,
    sub_expr: str,
    job_name: str,
    dead_letter_topic: str,
) -> list:
    """Build workflow steps for a continuation stage's pull-run-ack loop.

    Flow: pull message → run job → based on outcome:
      - Success (exit 0): ACK immediately, proceed
      - Reject (exit 2): ACK immediately, loop back to pull next message
      - Exception (exit != 0,2): retry up to MAX_EXCEPTION_RETRIES, then dead-letter
    """
    s = stage_index
    return [
        {
            f"pull_stage_{s}": {
                "call": "googleapis.pubsub.v1.projects.subscriptions.pull",
                "args": {
                    "subscription": sub_expr,
                    "body": {"maxMessages": 1},
                },
                "result": "pull_response",
            },
        },
        {
            f"check_message_{s}": {
                "switch": [{
                    "condition": '${not("receivedMessages" in pull_response.body) or len(pull_response.body.receivedMessages) == 0}',
                    "steps": [
                        {"sleep_backoff": {"call": "sys.sleep", "args": {"seconds": 30}}},
                        {"retry_pull": {"next": f"pull_stage_{s}"}},
                    ],
                }],
            },
        },
        {
            f"decode_event_{s}": {
                "assign": [
                    {"current_ack_id": "${pull_response.body.receivedMessages[0].ackId}"},
                    {"stage_event": "${json.decode(base64.decode(pull_response.body.receivedMessages[0].message.data))}"},
                ],
            },
        },
        {
            f"run_stage_{s}": {
                "try": {
                    "steps": [{
                        f"execute_job_{s}": {
                            "call": "googleapis.run.v2.projects.locations.jobs.run",
                            "args": {
                                "name": job_name,
                                "body": {
                                    "overrides": {
                                        "containerOverrides": [{"env": [
                                            {"name": "LAMIA_TRIGGER_EVENT", "value": "${json.encode(stage_event)}"},
                                            {"name": "LAMIA_STAGE_CONTEXT", "value": "${json.encode(stage_context)}"},
                                        ]}],
                                    },
                                },
                            },
                            "result": f"stage_{s}_result",
                        },
                    }],
                    "except": {
                        "as": "job_error",
                        "steps": [
                            {
                                f"check_reject_{s}": {
                                    "switch": [{
                                        "condition": f'${{\"exitCode\" in job_error and job_error.exitCode == {EXIT_CODE_REJECT}}}',
                                        "steps": [
                                            {
                                                "ack_rejected": {
                                                    "call": "googleapis.pubsub.v1.projects.subscriptions.acknowledge",
                                                    "args": {
                                                        "subscription": sub_expr,
                                                        "body": {"ackIds": ["${current_ack_id}"]},
                                                    },
                                                },
                                            },
                                            {"loop_after_reject": {"next": f"pull_stage_{s}"}},
                                        ],
                                    }],
                                },
                            },
                            {
                                f"handle_exception_{s}": {
                                    "assign": [{"retry_count": "${retry_count + 1}"}],
                                },
                            },
                            {
                                f"check_max_retries_{s}": {
                                    "switch": [{
                                        "condition": f"${{retry_count >= {MAX_EXCEPTION_RETRIES}}}",
                                        "steps": [
                                            {
                                                "ack_exhausted": {
                                                    "call": "googleapis.pubsub.v1.projects.subscriptions.acknowledge",
                                                    "args": {
                                                        "subscription": sub_expr,
                                                        "body": {"ackIds": ["${current_ack_id}"]},
                                                    },
                                                },
                                            },
                                            {
                                                "publish_dead_letter": {
                                                    "call": "googleapis.pubsub.v1.projects.topics.publish",
                                                    "args": {
                                                        "topic": dead_letter_topic,
                                                        "body": {
                                                            "messages": [{"data": "${base64.encode(json.encode(stage_event))}"}],
                                                        },
                                                    },
                                                },
                                            },
                                            {
                                                "raise_dead_letter": {
                                                    "raise": "${\"Event dead-lettered after \" + string(retry_count) + \" retries: \" + job_error.message}",
                                                },
                                            },
                                        ],
                                    }],
                                },
                            },
                            {"retry_after_error": {"next": f"pull_stage_{s}"}},
                        ],
                    },
                },
            },
        },
        {
            f"ack_success_{s}": {
                "call": "googleapis.pubsub.v1.projects.subscriptions.acknowledge",
                "args": {
                    "subscription": sub_expr,
                    "body": {"ackIds": ["${current_ack_id}"]},
                },
            },
        },
    ]


def _multi_stage_workflow(plan: TriggerDeploymentPlan, project_id: str, location: str) -> str:
    """Multi-stage workflow with per-execution isolated subscriptions.

    Each execution creates its own subscription on the continuation topic,
    ensuring that concurrent executions never steal each other's events.
    Ack/nack is always sent AFTER job completes, never before.
    """
    total = len(plan.stages)
    steps = []

    steps.append({
        "init_execution_id": {
            "assign": [{
                "exec_id": "${sys.get_env(\"GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID\")}",
            }],
        },
    })

    stage_0_job = _job_resource_name(project_id, location, plan.name, 0, total)
    steps.append({
        "run_stage_0": {
            "call": "googleapis.run.v2.projects.locations.jobs.run",
            "args": {
                "name": stage_0_job,
                "body": {
                    "overrides": {
                        "containerOverrides": [{"env": [
                            {"name": "LAMIA_TRIGGER_EVENT", "value": "${json.encode(event)}"},
                        ]}],
                    },
                },
            },
            "result": "stage_0_result",
        },
    })

    for i in range(total - 1):
        steps.append({
            f"update_context_{i}": {
                "assign": [{"stage_context": f"${{stage_{i}_result}}"}],
            },
        })

        topic = f"projects/{project_id}/topics/lamia-trigger-{plan.name}-stage-{i + 1}"
        sub_base = f"projects/{project_id}/subscriptions/lamia-trigger-{plan.name}-stage-{i + 1}"
        dead_letter_topic = f"projects/{project_id}/topics/lamia-trigger-{plan.name}-dead-letter"

        steps.append({
            f"create_sub_stage_{i + 1}": {
                "call": "googleapis.pubsub.v1.projects.subscriptions.create",
                "args": {
                    "name": f"{sub_base}-${{exec_id}}",
                    "body": {
                        "topic": topic,
                        "ackDeadlineSeconds": 600,
                        "expirationPolicy": {
                            "ttl": f"{CONTINUATION_TIMEOUT_SECONDS + 3600}s",
                        },
                    },
                },
                "result": f"sub_stage_{i + 1}",
            },
        })

        steps.append({
            f"init_retry_stage_{i + 1}": {
                "assign": [{"retry_count": 0}],
            },
        })

        next_job = _job_resource_name(project_id, location, plan.name, i + 1, total)
        continuation_steps = _build_continuation_loop(
            stage_index=i + 1,
            sub_expr=f"{sub_base}-${{exec_id}}",
            job_name=next_job,
            dead_letter_topic=dead_letter_topic,
        )
        steps.extend(continuation_steps)

        steps.append({
            f"cleanup_sub_stage_{i + 1}": {
                "call": "googleapis.pubsub.v1.projects.subscriptions.delete",
                "args": {
                    "subscription": f"{sub_base}-${{exec_id}}",
                },
            },
        })

    steps.append({
        "return_final": {
            "return": f"${{stage_{total - 1}_result}}",
        },
    })

    workflow = {
        "main": {
            "params": ["event"],
            "steps": steps,
        },
    }
    return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
