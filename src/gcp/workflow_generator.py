"""Generate Google Cloud Workflow YAML from a TriggerDeploymentPlan.

Single-trigger: Eventarc starts workflow, workflow runs Job with env override.
Multi-trigger: After each Job, workflow waits via Pub/Sub pull for next event.
"""

import json
import yaml
from typing import Optional

from lamia_cloud.types import TriggerDeploymentPlan, TriggerStage


def generate_workflow_yaml(
    plan: TriggerDeploymentPlan,
    project_id: str,
    location: str,
) -> str:
    """Generate Workflow definition YAML for the given trigger plan."""
    if len(plan.stages) == 1:
        return _single_stage_workflow(plan, project_id, location)
    return _multi_stage_workflow(plan, project_id, location)


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
    total = len(plan.stages)
    steps = []

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

            subscription = f"projects/{project_id}/subscriptions/lamia-trigger-{plan.name}-stage-{i + 1}"
            steps.append({
                f"wait_for_stage_{i + 1}": {
                    "call": "googleapis.pubsub.v1.projects.subscriptions.pull",
                    "args": {
                        "subscription": subscription,
                        "body": {"maxMessages": 1},
                    },
                    "result": "pull_response",
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
                        "subscription": subscription,
                        "body": {
                            "ackIds": ["${pull_response.body.receivedMessages[0].ackId}"],
                        },
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
