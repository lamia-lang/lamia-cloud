"""Generate Google Cloud Workflow YAML from a TriggerDeploymentPlan.

Single-trigger reactive: Eventarc starts workflow, workflow runs Job with env override.
Multi-trigger reactive: After each Job, workflow waits via isolated per-execution
    Pub/Sub subscription for the next continuation event.
Drain (scheduled): Pulls all accumulated events from Pub/Sub, fans out one Job per event.
"""

import json
import yaml
from typing import Optional

from lamia_cloud.types import TriggerDeploymentPlan, TriggerStage

CONTINUATION_TIMEOUT_SECONDS = 259200  # 72 hours


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
    """Build workflow steps that process each message in the pulled batch."""
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
                        },
                    ],
                },
            },
        },
    })

    steps.append({
        "ack_batch": {
            "call": "googleapis.pubsub.v1.projects.subscriptions.acknowledge",
            "args": {
                "subscription": subscription,
                "body": {
                    "ackIds": "${[msg.ackId for msg in pull_response.body.receivedMessages]}",
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
    job_name = _job_resource_name(project_id, location, plan.name, 0, 1)
    workflow = {
        "main": {
            "params": ["event"],
            "steps": [
                {
                    "run_stage": {
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


def _multi_stage_workflow(plan: TriggerDeploymentPlan, project_id: str, location: str) -> str:
    """Multi-stage workflow with per-execution isolated subscriptions.

    Each execution creates its own subscription on the continuation topic,
    ensuring that concurrent executions never steal each other's events.
    """
    total = len(plan.stages)
    steps = []

    # Derive a unique execution suffix from the workflow execution ID
    steps.append({
        "init_execution_id": {
            "assign": [{
                "exec_id": "${sys.get_env(\"GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID\")}",
            }],
        },
    })

    for i, stage in enumerate(plan.stages):
        job_name = _job_resource_name(project_id, location, plan.name, i, total)

        if i == 0:
            event_expr = "${json.encode(event)}"
        else:
            event_expr = "${json.encode(stage_event)}"

        env_vars = [{"name": "LAMIA_TRIGGER_EVENT", "value": event_expr}]
        if i > 0:
            env_vars.append({
                "name": "LAMIA_STAGE_CONTEXT",
                "value": "${json.encode(stage_context)}",
            })

        steps.append({
            f"run_stage_{i}": {
                "call": "googleapis.run.v2.projects.locations.jobs.run",
                "args": {
                    "name": job_name,
                    "body": {
                        "overrides": {
                            "containerOverrides": [{"env": env_vars}],
                        },
                    },
                },
                "result": f"stage_{i}_result",
            },
        })

        if i < total - 1:
            steps.append({
                f"update_context_{i}": {
                    "assign": [{
                        "stage_context": f"${{stage_{i}_result}}",
                    }],
                },
            })

            topic = f"projects/{project_id}/topics/lamia-trigger-{plan.name}-stage-{i + 1}"
            sub_base = f"projects/{project_id}/subscriptions/lamia-trigger-{plan.name}-stage-{i + 1}"

            # Create per-execution subscription for isolation
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

            # Poll loop: pull until a message arrives or timeout
            steps.append({
                f"wait_for_stage_{i + 1}": {
                    "try": {
                        "steps": [{
                            f"pull_stage_{i + 1}": {
                                "call": "googleapis.pubsub.v1.projects.subscriptions.pull",
                                "args": {
                                    "subscription": f"{sub_base}-${{exec_id}}",
                                    "body": {"maxMessages": 1},
                                },
                                "result": "pull_response",
                            },
                        }],
                        "retry": {
                            "predicate": "${not(\"receivedMessages\" in pull_response.body) or len(pull_response.body.receivedMessages) == 0}",
                            "max_retries": 8640,
                            "backoff": {
                                "initial_delay": 30,
                                "max_delay": 30,
                                "multiplier": 1,
                            },
                        },
                    },
                },
            })

            steps.append({
                f"decode_stage_{i + 1}_event": {
                    "assign": [{
                        "stage_event": "${json.decode(base64.decode(pull_response.body.receivedMessages[0].message.data))}",
                    }],
                },
            })

            steps.append({
                f"ack_stage_{i + 1}": {
                    "call": "googleapis.pubsub.v1.projects.subscriptions.acknowledge",
                    "args": {
                        "subscription": f"{sub_base}-${{exec_id}}",
                        "body": {
                            "ackIds": ["${pull_response.body.receivedMessages[0].ackId}"],
                        },
                    },
                },
            })

            # Cleanup: delete the per-execution subscription
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
