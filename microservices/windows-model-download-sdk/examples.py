"""
Integration tests and usage examples for Phase 2 implementation.

This file demonstrates the ModelDownloadClient wrapper in action with
realistic usage patterns.
"""

import asyncio
from typing import List

from model_download_sdk import (
    ModelDownloadSDK,
    ModelSpec,
    ModelHub,
    ModelType,
    Job,
    JobStatus,
)
from model_download_sdk.exceptions import (
    SDKConnectionError,
    ValidationError,
    NotFoundError,
)


async def example_health_check() -> None:
    """Example 1: Check service health."""
    print("\n=== Example 1: Health Check ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        health = await client.health_check()
        print(f"Service health: {health}")
    except SDKConnectionError as e:
        print(f"Service unavailable: {e}")
    finally:
        await client.close()


async def example_list_plugins() -> None:
    """Example 2: List available plugins."""
    print("\n=== Example 2: List Plugins ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        plugins = await client.list_plugins()
        print(f"Available plugins: {len(plugins)}")
        for plugin in plugins[:3]:
            print(f"  - {plugin.get('name', 'unknown')} v{plugin.get('version', '?')}")
    except SDKConnectionError as e:
        print(f"Failed to list plugins: {e}")
    finally:
        await client.close()


async def example_download_single_model() -> None:
    """Example 3: Download a single model."""
    print("\n=== Example 3: Download Single Model ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        # Create model spec
        spec = ModelSpec(
            name="microsoft/Phi-3.5-mini-instruct",
            hub=ModelHub.HUGGINGFACE,
            type_=ModelType.LLM,
            convert_to_openvino=False,
        )
        print(f"Downloading: {spec.name}")
        
        # Initiate download
        result = await client.download_models(
            models=[spec],
            output_directory="/tmp/models",
            wait=False,  # Don't wait for completion
        )
        
        print(f"Download initiated:")
        print(f"  - Job IDs: {result.job_ids}")
        print(f"  - Output: {result.output_directory}")
        
        # Check job status
        if result.job_ids:
            job_id = result.job_ids[0]
            job = await client.get_job(job_id)
            print(f"  - Job status: {job.status}")
            
    except ValidationError as e:
        print(f"Validation error: {e}")
    except SDKConnectionError as e:
        print(f"Connection error: {e}")
    finally:
        await client.close()


async def example_wait_for_job() -> None:
    """Example 4: Download and wait for job completion."""
    print("\n=== Example 4: Download with Job Polling ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        spec = ModelSpec(
            name="gpt2",
            hub=ModelHub.HUGGINGFACE,
            type_=ModelType.LLM,
        )
        
        # Download and wait (simulated - server may not actually download)
        result = await client.download_models(
            models=[spec],
            output_directory="/tmp/models",
            wait=True,
            timeout=10,  # 10 second timeout for demo
        )
        
        print(f"Download completed:")
        print(f"  - Successful: {len(result.successful_jobs)}")
        print(f"  - Failed: {len(result.failed_jobs)}")
        
    except Exception as e:
        print(f"Note: {type(e).__name__}: {e}")
    finally:
        await client.close()


async def example_list_jobs() -> None:
    """Example 5: List all jobs with filtering."""
    print("\n=== Example 5: List Jobs ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        # List all jobs
        all_jobs = await client.list_jobs()
        print(f"Total jobs: {len(all_jobs)}")
        
        # Filter by status
        pending = await client.list_jobs(status=JobStatus.PENDING)
        print(f"  - Pending: {len(pending)}")
        
        processing = await client.list_jobs(status=JobStatus.PROCESSING)
        print(f"  - Processing: {len(processing)}")
        
        # Show first few jobs
        for job in all_jobs[:3]:
            print(f"  - {job.id}: {job.model_name} ({job.status})")
            
    except SDKConnectionError as e:
        print(f"Failed to list jobs: {e}")
    finally:
        await client.close()


async def example_error_handling() -> None:
    """Example 6: Error handling scenarios."""
    print("\n=== Example 6: Error Handling ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    
    # Test 1: Invalid job ID
    try:
        print("Test 1: Getting non-existent job...")
        await client.get_job("invalid-job-12345")
        print("Should have raised NotFoundError")
    except NotFoundError as e:
        print(f"Correctly raised NotFoundError: {e}")
    except Exception as e:
        print(f"? Got {type(e).__name__}: {e}")
    
    # Test 2: Empty model list
    try:
        print("\nTest 2: Downloading empty model list...")
        await client.download_models(
            models=[],
            output_directory="/tmp/models",
        )
        print("Should have raised ValidationError")
    except ValidationError as e:
        print(f"Correctly raised ValidationError: {e}")
    except Exception as e:
        print(f"? Got {type(e).__name__}: {e}")
    
    # Test 3: Invalid model spec
    try:
        print("\nTest 3: Creating invalid model spec...")
        spec = ModelSpec(
            name="test",
            hub="invalid-hub",  # Invalid hub
            type_=ModelType.LLM,
        )
        print("Should have raised ValidationError")
    except ValidationError as e:
        print(f"Correctly raised ValidationError: {e}")
    except Exception as e:
        print(f"? Got {type(e).__name__}: {e}")
    
    await client.close()


async def example_context_manager() -> None:
    """Example 7: Using as async context manager."""
    print("\n=== Example 7: Async Context Manager ===")
    
    try:
        async with ModelDownloadSDK(base_url="http://localhost:8200") as client:
            plugins = await client.list_plugins()
            print(f"Listed {len(plugins)} plugins within context")
        print("Context manager cleaned up properly")
    except Exception as e:
        print(f"Note: {e}")


async def example_configuration() -> None:
    """Example 8: Custom SDK configuration."""
    print("\n=== Example 8: Custom Configuration ===")
    
    from model_download_sdk.client import SDKConfig
    
    config = SDKConfig(
        base_url="https://api.example.com:8200",
        timeout=60.0,
        verify_ssl=True,
        job_poll_interval=1.0,  # Poll every 1 second
        job_max_timeout=3600,   # Max 1 hour wait
        normalize_paths=True,
        validate_paths=True,
    )
    
    client = ModelDownloadSDK(config=config)
    print(f"Configured SDK:")
    print(f"  - Base URL: {config.base_url}")
    print(f"  - Timeout: {config.timeout}s")
    print(f"  - Poll interval: {config.job_poll_interval}s")
    print(f"  - Max job timeout: {config.job_max_timeout}s")
    
    await client.close()


async def example_download_single_model_simple() -> None:
    """Example 8: Download single model with simple string API."""
    print("\n=== Example 8: Download Single Model (Simple String API) ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        # Simple string-based API (new feature!)
        result = await client.download_model(
            model_name="microsoft/Phi-3.5-mini-instruct",
            hub="huggingface",
            download_path="models",
            model_type="llm",
            convert_to_openvino=True,
        )
        
        print(f"Download initiated:")
        print(f"  - Job IDs: {result.job_ids}")
        print(f"  - Output: {result.output_directory}")
        
        # The simple API still returns strongly typed SDK objects
        print(f"  - Result type: {type(result).__name__}")
        
    except ValidationError as e:
        print(f"Validation error: {e}")
    except SDKConnectionError as e:
        print(f"Connection error: {e}")
    finally:
        await client.close()


async def example_download_model_with_pathlib() -> None:
    """Example 9: Download model with pathlib path handling."""
    print("\n=== Example 9: Download with Path Handling ===")
    
    from pathlib import Path
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        # Can use Path objects or strings - internally uses pathlib
        download_dir = Path("models") / "language-models"
        
        result = await client.download_model(
            model_name="gpt2",
            hub="huggingface",
            download_path=str(download_dir),
            wait=False,
        )
        
        print(f"Download started:")
        print(f"  - Model: gpt2")
        print(f"  - Download path: {download_dir}")
        print(f"  - Job created: {result.job_ids[0] if result.job_ids else 'N/A'}")
        
    except Exception as e:
        print(f"Note: {type(e).__name__}: {e}")
    finally:
        await client.close()


async def example_download_model_hub_aliases() -> None:
    """Example 10: Download model using hub aliases."""
    print("\n=== Example 10: Hub Aliases ===")
    
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    try:
        # Supported hub aliases: huggingface/hf, ollama, yolo/ultralytics, geti, openvino, hls, pipeline-zoo
        
        print("Hub aliases are supported:")
        print("  - 'huggingface' or 'hf' → HuggingFace")
        print("  - 'ollama' → Ollama")
        print("  - 'yolo' or 'ultralytics' → Ultralytics")
        print("  - 'geti' → Intel Geti")
        print("  - 'openvino' → OpenVINO")
        print("  - 'hls' → Custom HLS")
        print("  - 'pipeline-zoo' → Pipeline Zoo")
        
        # Try with alias
        print("\nTrying download with 'hf' alias:")
        result = await client.download_model(
            model_name="bert-base-uncased",
            hub="hf",  # Using alias instead of "huggingface"
            download_path="models",
            wait=False,
        )
        print(f"Download initiated using 'hf' alias")
        
    except Exception as e:
        print(f"Note: {type(e).__name__}: {e}")
    finally:
        await client.close()


async def run_all_examples() -> None:
    """Run all examples."""
    print("=" * 60)
    print("PHASE 2+ IMPLEMENTATION: SDK CLIENT EXAMPLES")
    print("=" * 60)
    
    await example_configuration()
    await example_health_check()
    await example_list_plugins()
    await example_download_single_model()
    await example_list_jobs()
    await example_error_handling()
    await example_context_manager()
    await example_download_single_model_simple()
    await example_download_model_with_pathlib()
    await example_download_model_hub_aliases()
    
    print("\n" + "=" * 60)
    print("Examples completed")
    print("=" * 60)


if __name__ == "__main__":
    # Note: Run with actual service running, or examples will show connection errors
    print("\nNote: These examples demonstrate SDK usage patterns.")
    print("They require a running Model Download Service instance.")
    print("If service is not running, you'll see connection errors.")
    print("\nTo run: python -m pytest tests/ or python examples.py")
    
    asyncio.run(run_all_examples())
