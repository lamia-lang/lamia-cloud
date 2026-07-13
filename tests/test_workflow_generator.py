"""Tests for lamia_cloud.gcp.workflow_generator's retry/ACK-NACK step generation.

_build_continuation_loop is pure — it returns a Python data structure describing
GCP Workflow steps, with no GCP calls — so these test the generated structure
directly rather than mocking anything.
"""

import json

from lamia_cloud.gcp.workflow_generator import (
    EXIT_CODE_REJECT,
    MAX_EXCEPTION_RETRIES,
    _build_continuation_loop,
)


class TestBuildContinuationLoop:
    def test_includes_reject_and_retry_exhaustion_logic(self):
        steps = _build_continuation_loop(
            stage_index=1,
            sub_expr="${continuation_sub}",
            job_name="projects/proj/locations/us-central1/jobs/my-job",
            dead_letter_topic="projects/proj/topics/my-trigger-dead-letter",
        )
        rendered = json.dumps(steps)

        assert f"exitCode == {EXIT_CODE_REJECT}" in rendered
        assert f"retry_count >= {MAX_EXCEPTION_RETRIES}" in rendered
        assert "publish_dead_letter" in rendered
        assert "ack_rejected" in rendered
        assert "ack_exhausted" in rendered

    def test_step_names_include_stage_index(self):
        steps = _build_continuation_loop(
            stage_index=2, sub_expr="${sub}", job_name="job", dead_letter_topic="topic",
        )
        step_keys = [list(s.keys())[0] for s in steps]
        assert "pull_stage_2" in step_keys
        assert "check_message_2" in step_keys
        assert "decode_event_2" in step_keys
        assert "run_stage_2" in step_keys
        assert "ack_success_2" in step_keys

    def test_pull_uses_given_subscription_expression(self):
        steps = _build_continuation_loop(
            stage_index=0, sub_expr="${my_sub}", job_name="job", dead_letter_topic="topic",
        )
        pull_step = steps[0]["pull_stage_0"]
        assert pull_step["args"]["subscription"] == "${my_sub}"

    def test_dead_letter_publish_targets_given_topic(self):
        steps = _build_continuation_loop(
            stage_index=0, sub_expr="${sub}", job_name="job",
            dead_letter_topic="projects/proj/topics/custom-dead-letter",
        )
        rendered = json.dumps(steps)
        assert "projects/proj/topics/custom-dead-letter" in rendered
