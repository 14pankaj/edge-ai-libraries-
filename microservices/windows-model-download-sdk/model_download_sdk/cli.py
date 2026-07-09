"""Typer-based CLI for the Model Download SDK.

Commands:
	model-download health
	model-download plugins
	model-download jobs
	model-download download
	model-download results
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from model_download_sdk.client import ModelDownloadSDK, SDKConfig
from model_download_sdk.exceptions import (
	AuthenticationError,
	JobError,
	NotFoundError,
	SDKError,
	TimeoutError,
	ValidationError,
)
from model_download_sdk.models import Job, JobStatus


app = typer.Typer(
	name="model-download",
	help="Model Download SDK CLI",
	add_completion=False,
)
console = Console()


def _configure_logging(verbose: bool) -> None:
	"""Configure SDK/CLI logging level."""
	level = logging.DEBUG if verbose else logging.WARNING
	logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _render_sdk_error(exc: SDKError, verbose: bool) -> None:
	"""Render SDK exceptions with rich formatting."""
	title = "SDK Error"
	details = [str(exc)]

	if isinstance(exc, ValidationError):
		title = "Validation Error"
		if exc.field:
			details.append(f"Field: {exc.field}")
		if exc.value is not None:
			details.append(f"Value: {exc.value}")
	elif isinstance(exc, AuthenticationError):
		title = "Authentication Error"
	elif isinstance(exc, NotFoundError):
		title = "Not Found"
		if exc.resource_type:
			details.append(f"Resource Type: {exc.resource_type}")
		if exc.resource_id:
			details.append(f"Resource ID: {exc.resource_id}")
	elif isinstance(exc, TimeoutError):
		title = "Timeout"
		details.append(f"Timeout Seconds: {exc.timeout_seconds}")
		if exc.operation:
			details.append(f"Operation: {exc.operation}")
	elif isinstance(exc, JobError):
		title = "Job Failed"
		details.append(f"Job ID: {exc.job_id}")
		if exc.status:
			details.append(f"Status: {exc.status}")

	body = "\n".join(details)
	console.print(Panel.fit(body, title=title, border_style="red"))
	if verbose:
		console.print_exception(show_locals=False)


def _status_from_string(status: Optional[str]) -> Optional[JobStatus]:
	"""Convert CLI status string to JobStatus enum."""
	if status is None:
		return None
	try:
		return JobStatus(status.lower())
	except ValueError as exc:
		valid = ", ".join(s.value for s in JobStatus)
		raise ValidationError(
			f"Invalid status '{status}'. Must be one of: {valid}",
			field="status",
			value=status,
		) from exc


def _print_jobs(jobs: list[Job], title: str) -> None:
	"""Render job list as rich table."""
	table = Table(title=title)
	table.add_column("Job ID", overflow="fold")
	table.add_column("Status")
	table.add_column("Operation")
	table.add_column("Model")
	table.add_column("Hub")
	table.add_column("Output")
	table.add_column("Error", overflow="fold")

	for job in jobs:
		table.add_row(
			str(job.id),
			str(job.status.value),
			str(job.operation.value),
			str(job.model_name),
			str(job.hub),
			str(job.output_directory) if job.output_directory else "-",
			str(job.error) if job.error else "-",
		)

	console.print(table)


def _run_with_client(
	*,
	base_url: str,
	timeout: float,
	verify_ssl: bool,
	verbose: bool,
	action: Callable[[ModelDownloadSDK], Awaitable[None]],
) -> None:
	"""Execute an async command with shared SDK setup/teardown."""
	_configure_logging(verbose)

	async def _runner() -> None:
		config = SDKConfig(
			base_url=base_url,
			timeout=timeout,
			verify_ssl=verify_ssl,
		)
		client = ModelDownloadSDK(config=config)
		try:
			await action(client)
		finally:
			await client.close()

	try:
		asyncio.run(_runner())
	except SDKError as exc:
		_render_sdk_error(exc, verbose=verbose)
		raise typer.Exit(code=1)
	except Exception as exc:  # pragma: no cover - defensive guard
		console.print(Panel.fit(str(exc), title="Unexpected Error", border_style="red"))
		if verbose:
			console.print_exception(show_locals=False)
		raise typer.Exit(code=1)


@app.command("health")
def health(
	base_url: str = typer.Option("http://localhost:8200", help="Model Download Service base URL."),
	timeout: float = typer.Option(30.0, help="HTTP timeout in seconds."),
	verify_ssl: bool = typer.Option(True, "--verify-ssl/--no-verify-ssl", help="Enable SSL verification."),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
) -> None:
	"""Check service health."""

	async def _action(client: ModelDownloadSDK) -> None:
		result = await client.health_check()
		console.print(Panel.fit("Service is reachable", title="Health", border_style="green"))
		console.print_json(data=json.dumps(result, default=str))

	_run_with_client(
		base_url=base_url,
		timeout=timeout,
		verify_ssl=verify_ssl,
		verbose=verbose,
		action=_action,
	)


@app.command("plugins")
def plugins(
	base_url: str = typer.Option("http://localhost:8200", help="Model Download Service base URL."),
	timeout: float = typer.Option(30.0, help="HTTP timeout in seconds."),
	verify_ssl: bool = typer.Option(True, "--verify-ssl/--no-verify-ssl", help="Enable SSL verification."),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
) -> None:
	"""List available plugins."""

	async def _action(client: ModelDownloadSDK) -> None:
		plugin_list = await client.list_plugins()
		if not plugin_list:
			console.print(Panel.fit("No plugins available.", title="Plugins", border_style="yellow"))
			return
		table = Table(title="Plugins")
		table.add_column("Name")
		table.add_column("Version")
		table.add_column("Details", overflow="fold")
		for plugin in plugin_list:
			table.add_row(
				str(plugin.get("name", "-")),
				str(plugin.get("version", "-")),
				json.dumps(plugin, default=str),
			)
		console.print(table)

	_run_with_client(
		base_url=base_url,
		timeout=timeout,
		verify_ssl=verify_ssl,
		verbose=verbose,
		action=_action,
	)


@app.command("jobs")
def jobs(
	status: Optional[str] = typer.Option(None, help="Filter status: pending|processing|completed|failed"),
	limit: int = typer.Option(100, min=1, help="Maximum number of jobs."),
	offset: int = typer.Option(0, min=0, help="Pagination offset."),
	base_url: str = typer.Option("http://localhost:8200", help="Model Download Service base URL."),
	timeout: float = typer.Option(30.0, help="HTTP timeout in seconds."),
	verify_ssl: bool = typer.Option(True, "--verify-ssl/--no-verify-ssl", help="Enable SSL verification."),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
) -> None:
	"""List jobs with optional filters."""

	async def _action(client: ModelDownloadSDK) -> None:
		status_filter = _status_from_string(status)
		job_list = await client.list_jobs(status=status_filter, limit=limit, offset=offset)
		_print_jobs(job_list, title="Jobs")

	_run_with_client(
		base_url=base_url,
		timeout=timeout,
		verify_ssl=verify_ssl,
		verbose=verbose,
		action=_action,
	)


@app.command("download")
def download(
	model_name: str = typer.Option(..., "--model-name", "-m", help="Model identifier."),
	hub: str = typer.Option(..., "--hub", help="Model hub, e.g., huggingface."),
	download_path: Path = typer.Option(Path("models"), "--download-path", help="Download directory."),
	model_type: Optional[str] = typer.Option(None, "--model-type", help="Model type."),
	revision: Optional[str] = typer.Option(None, "--revision", help="Model revision/tag."),
	convert_to_openvino: bool = typer.Option(False, "--convert-to-openvino/--no-convert-to-openvino", help="Convert model to OpenVINO IR."),
	wait: bool = typer.Option(False, "--wait/--no-wait", help="Wait for completion before returning."),
	job_timeout: Optional[int] = typer.Option(None, "--job-timeout", min=1, help="Job wait timeout in seconds."),
	base_url: str = typer.Option("http://localhost:8200", help="Model Download Service base URL."),
	timeout: float = typer.Option(30.0, help="HTTP timeout in seconds."),
	verify_ssl: bool = typer.Option(True, "--verify-ssl/--no-verify-ssl", help="Enable SSL verification."),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
) -> None:
	"""Download a model using SDK workflow."""

	async def _action(client: ModelDownloadSDK) -> None:
		result = await client.download_model(
			model_name=model_name,
			hub=hub,
			download_path=str(download_path),
			model_type=model_type,
			convert_to_openvino=convert_to_openvino,
			revision=revision,
			wait=wait,
			timeout=job_timeout,
		)

		console.print(
			Panel.fit(
				f"Submitted {len(result.job_ids)} job(s)\nOutput: {result.output_directory}",
				title="Download",
				border_style="green",
			)
		)

		if result.job_ids:
			table = Table(title="Created Jobs")
			table.add_column("Job ID")
			for job_id in result.job_ids:
				table.add_row(job_id)
			console.print(table)

		if wait:
			if result.successful_jobs:
				_print_jobs(result.successful_jobs, title="Successful Jobs")
			if result.failed_jobs:
				_print_jobs(result.failed_jobs, title="Failed Jobs")

	_run_with_client(
		base_url=base_url,
		timeout=timeout,
		verify_ssl=verify_ssl,
		verbose=verbose,
		action=_action,
	)


@app.command("results")
def results(
	limit: int = typer.Option(100, min=1, help="Maximum number of results."),
	offset: int = typer.Option(0, min=0, help="Pagination offset."),
	base_url: str = typer.Option("http://localhost:8200", help="Model Download Service base URL."),
	timeout: float = typer.Option(30.0, help="HTTP timeout in seconds."),
	verify_ssl: bool = typer.Option(True, "--verify-ssl/--no-verify-ssl", help="Enable SSL verification."),
	verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
) -> None:
	"""Get completed model results."""

	async def _action(client: ModelDownloadSDK) -> None:
		result_jobs = await client.get_model_results(limit=limit, offset=offset)
		_print_jobs(result_jobs, title="Model Results")

	_run_with_client(
		base_url=base_url,
		timeout=timeout,
		verify_ssl=verify_ssl,
		verbose=verbose,
		action=_action,
	)


def main() -> None:
	"""CLI entrypoint."""
	app()


if __name__ == "__main__":
	main()
