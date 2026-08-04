"""Regression tests for generated client job status deserialization."""

from __future__ import annotations


def test_generated_job_deserializes_downloading_status() -> None:
    """Generated Job.from_dict should accept runtime status='downloading'."""
    from model_download_service_api_client.models.job import Job
    from model_download_service_api_client.models.job_status import JobStatus

    payload = {
        "job_id": "job-123",
        "operation_type": "download",
        "model_name": "bert-base-uncased",
        "hub": "huggingface",
        "status": "downloading",
    }

    job = Job.from_dict(payload)

    assert job.status == JobStatus.DOWNLOADING
    assert str(job.status.value) == "downloading"
