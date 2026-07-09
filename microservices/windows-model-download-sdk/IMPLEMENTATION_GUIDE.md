# Model Download SDK - Phase 2 Implementation Guide

## Overview

The Model Download SDK (Phase 2) provides a complete implementation of the ModelDownloadClient wrapper layer that bridges the SDK and the generated API client. All 6 endpoints are fully functional with comprehensive error handling, logging, and type hints.

## Architecture

### Layer Structure

```
┌─────────────────────────────────────────┐
│  User Code (SDK API)                    │
│  - ModelDownloadSDK class               │
│  - SDK models (ModelSpec, Job, etc.)    │
│  - SDK exceptions                       │
└─────────────────────────────────────────┘
              ↓ (async methods)
┌─────────────────────────────────────────┐
│  HTTP Client Layer (_http_client.py)    │
│  - ModelDownloadClient class            │
│  - Wraps generated client               │
│  - Handles errors & logging             │
└─────────────────────────────────────────┘
              ↓ (sync methods)
┌─────────────────────────────────────────┐
│  Adapter Layer (_generated_adapter.py)  │
│  - GeneratedClientWrapper               │
│  - Model conversions (SDK ↔ Generated)  │
│  - Error message extraction             │
└─────────────────────────────────────────┘
              ↓ (imports)
┌─────────────────────────────────────────┐
│  Generated Client                       │
│  (model-download-service-api-client)    │
│  - Auto-generated from OpenAPI          │
│  - httpx-based HTTP client              │
│  - Generated models and endpoints       │
└─────────────────────────────────────────┘
```

### Key Design Principles

1. **Generation Isolation**: Generated code is never imported in public API
   - All generated imports in `_generated_adapter.py` only
   - Users only see SDK types (ModelSpec, Job, etc.)
   - Easy to regenerate without breaking SDK

2. **Async-First**: All SDK methods are async
   - Sync HTTP calls wrapped with `run_in_executor`
   - Supports async context manager (`async with`)
   - Proper asyncio integration

3. **Error Handling**: Comprehensive exception mapping
   - HTTP errors → SDK exception hierarchy
   - Context preserved (job_id, field names, etc.)
   - Consistent error messages

4. **Logging**: Strategic logging at appropriate levels
   - DEBUG: Operation flow details
   - INFO: Success milestones
   - ERROR: Failures with context

## Usage Examples

### Basic Usage

```python
import asyncio
from model_download_sdk import ModelDownloadSDK, ModelSpec, ModelHub, ModelType

async def main():
    # Create client
    client = ModelDownloadSDK(base_url="http://localhost:8200")
    
    try:
        # Check health
        health = await client.health_check()
        print(f"Service: {health['status']}")
        
        # Create model spec
        spec = ModelSpec(
            name="gpt2",
            hub=ModelHub.HUGGINGFACE,
            type_=ModelType.LLM,
        )
        
        # Download models
        result = await client.download_models(
            models=[spec],
            output_directory="~/models",
            wait=True,  # Wait for completion
            timeout=3600,
        )
        
        print(f"Success: {len(result.successful_jobs)}")
        print(f"Failed: {len(result.failed_jobs)}")
        
    finally:
        await client.close()

asyncio.run(main())
```

### Using Context Manager

```python
async def main():
    async with ModelDownloadSDK() as client:
        result = await client.download_models(...)
        # Client automatically closed when exiting context
```

### Custom Configuration

```python
from model_download_sdk.client import SDKConfig

config = SDKConfig(
    base_url="https://api.example.com:8200",
    timeout=60.0,
    verify_ssl=True,
    job_poll_interval=1.0,    # Poll every 1 second
    job_max_timeout=3600,     # Max 1 hour wait
    normalize_paths=True,      # Handle ~ and Windows long paths
    validate_paths=True,       # Validate before sending
)

client = ModelDownloadSDK(config=config)
```

### Job Polling

```python
async def main():
    client = ModelDownloadSDK()
    
    # Initiate download without waiting
    result = await client.download_models(
        models=[spec],
        output_directory="~/models",
        wait=False,  # Return immediately
    )
    
    # Later, wait for specific job
    job = await client.wait_for_job(
        job_id=result.job_ids[0],
        timeout=3600,
    )
    
    if job.is_success:
        print(f"✓ Downloaded to {job.output_directory}")
    else:
        print(f"✗ Failed: {job.error}")
    
    await client.close()
```

### Error Handling

```python
from model_download_sdk.exceptions import (
    ValidationError,
    SDKConnectionError,
    NotFoundError,
    TimeoutError,
)

try:
    result = await client.download_models(...)
except ValidationError as e:
    # Invalid input (bad model spec, invalid path, etc.)
    print(f"Validation error: {e}")
except SDKConnectionError as e:
    # Network error (timeout, connection failed, etc.)
    print(f"Connection error: {e}")
except NotFoundError as e:
    # Job not found
    print(f"Job not found: {e}")
except TimeoutError as e:
    # Operation exceeded timeout
    print(f"Timeout: {e}")
```

## Module Reference

### `client.py` - Main SDK Client

**Classes**:
- `SDKConfig`: Configuration options
- `ModelDownloadSDK`: Main async client

**Methods** (all async):
- `health_check()` - Check service health
- `download_models()` - Download and optionally convert models
- `download_model()` - Download single model (simple string API)
- `upload_model()` - Upload custom model (Phase 3+)
- `get_job()` - Get job status
- `list_jobs()` - List all jobs with filtering
- `wait_for_job()` - Poll until job completes
- `get_model_results()` - Get completed operations
- `list_plugins()` - List available plugins
- `close()` - Clean up resources

### `_http_client.py` - Low-Level HTTP Client

**Class**: `ModelDownloadClient`

**Methods** (all sync, called via `run_in_executor`):
- `health_check()` - Verify service
- `download_models()` - Request downloads
- `get_job_status()` - Get job status
- `list_jobs()` - List all jobs
- `get_model_results()` - Get results
- `list_plugins()` - List plugins

### `_generated_adapter.py` - Adapter Layer

**Functions**:
- `sdk_model_spec_to_generated()` - Convert SDK ModelSpec → Generated ModelRequest
- `generated_job_to_sdk()` - Convert Generated Job → SDK Job
- `generated_download_response_to_job_ids()` - Extract job IDs
- `generated_error_response_to_message()` - Extract error message
- `GeneratedClientWrapper` - Wrapper for generated endpoints

### `error_mapper.py` - Error Handling

**Functions**:
- `map_http_error()` - Map HTTP status → SDK exception
- `map_network_error()` - Map network error → SDKConnectionError
- `map_generated_error()` - Map generated exception → SDK exception
- `extract_error_message()` - Extract message from error response

### `types.py` - Type Definitions

**Classes**:
- `_Unset` - Marker for unset values
- `UNSET` - Singleton instance
- `Response[T]` - Generic response wrapper

## Implementation Details

### Download Models

```python
async def download_models(
    models: List[ModelSpec],
    output_directory: str,
    wait: bool = False,
    timeout: Optional[int] = None,
) -> DownloadResult:
    # 1. Validate inputs (non-empty models list)
    # 2. Normalize path (handle ~ and Windows long paths)
    # 3. Convert SDK models to generated format
    # 4. Call HTTP client endpoint
    # 5. Extract job IDs from response
    # 6. If wait=True:
    #    - Poll each job until completion
    #    - Collect successful and failed jobs
    #    - Respect timeout setting
    # 7. Return DownloadResult with job info
    # Raises: ValidationError, SDKConnectionError, TimeoutError
```

### Download Single Model (Simple String API)

```python
async def download_model(
    model_name: str,
    hub: str,
    download_path: str = ".",
    model_type: Optional[str] = None,
    convert_to_openvino: bool = False,
    revision: Optional[str] = None,
    wait: bool = False,
    timeout: Optional[int] = None,
) -> DownloadResult:
    # 1. Validate inputs (non-empty strings, valid hub/type)
    # 2. Convert hub/model_type strings to enums (supports aliases)
    # 3. Use pathlib.Path for path handling (cross-platform)
    # 4. Create strongly typed ModelSpec object
    # 5. Call download_models() with single model
    # 6. Return strongly typed DownloadResult
    # Raises: ValidationError, SDKConnectionError, TimeoutError
```

**Hub Aliases**: `"hf"` → HUGGINGFACE, `"yolo"` → ULTRALYTICS, etc.  
**Type Aliases**: `"llm"` → LLM, `"embed"` → EMBEDDINGS, `"image"` → VISION, etc.

Example:
```python
result = await client.download_model(
    model_name="gpt2",
    hub="hf",  # Alias for huggingface
    download_path="models",
    model_type="llm",  # Alias support
    wait=True,
)
```

### Health Check

```python
async def health_check(self) -> Dict[str, Any]:
    # Calls HTTP client
    # Returns: {"status": "...", "message": "..."}
    # Raises: SDKConnectionError
```

### Download Models

```python
async def download_models(
    models: List[ModelSpec],
    output_directory: str,
    wait: bool = False,
    timeout: Optional[int] = None,
) -> DownloadResult:
    # 1. Validate inputs (non-empty models list)
    # 2. Normalize path (handle ~ and Windows long paths)
    # 3. Convert SDK models to generated format
    # 4. Call HTTP client endpoint
    # 5. Extract job IDs from response
    # 6. If wait=True:
    #    - Poll each job until completion
    #    - Collect successful and failed jobs
    #    - Respect timeout setting
    # 7. Return DownloadResult with job info
    # Raises: ValidationError, SDKConnectionError, TimeoutError
```

### Job Polling

```python
async def wait_for_job(job_id: str, timeout: int) -> Job:
    # 1. Validate job_id (not empty)
    # 2. Loop until job completes or timeout:
    #    a. Call get_job(job_id)
    #    b. If complete: return Job
    #    c. If timeout exceeded: raise TimeoutError
    #    d. Sleep for poll_interval
    #    e. Check elapsed time
    # 3. Returns completed Job
    # Raises: TimeoutError, NotFoundError, SDKConnectionError
```

## Error Mapping

### HTTP Status → SDK Exception

| HTTP Status | SDK Exception | Reason |
|-------------|---------------|--------|
| 400 | ValidationError | Bad request |
| 401 | AuthenticationError | Unauthorized |
| 403 | AuthenticationError | Forbidden |
| 404 | NotFoundError | Resource not found |
| 422 | ValidationError | Invalid parameters |
| 5xx | SDKConnectionError | Server error |
| timeout | SDKConnectionError | Request timeout |
| connect error | SDKConnectionError | Connection failed |

## Logging Strategy

### DEBUG Level
- Client initialization
- Method calls with parameters
- API endpoint calls
- Response parsing

### INFO Level
- Service health status
- Download initiated (with job count)
- Job polling results
- Operation complete

### ERROR Level
- Connection failures
- Validation errors
- API errors (with status code)
- Timeout errors

### Example Log Output

```
DEBUG: Initializing ModelDownloadClient: base_url=http://localhost:8200, timeout=30s
INFO: ModelDownloadClient initialized successfully
DEBUG: Calling download_models: 2 models, path=/tmp/models
DEBUG: Converted 2 models to API format
INFO: Download initiated: 2 jobs created
DEBUG: Job job-123 status: processing
DEBUG: Job job-123 status: processing
DEBUG: Job job-123 status: completed
INFO: Download complete: 1 successful, 0 failed
```

## Path Handling

### Features
- Home directory expansion: `~/models` → `/home/user/models`
- Windows long path support: `\\?\C:\very\long\path` for 260+ chars
- Symlink resolution: Resolves symlinks to real paths
- UNC path support: `\\server\share\path`

### Configuration
```python
config = SDKConfig(
    normalize_paths=True,   # Auto-normalize paths
    validate_paths=True,    # Validate writable before sending
)
```

## Testing

### Unit Test Structure
```
tests/
├── test_adapter.py          # Model conversions
├── test_error_mapper.py     # Error handling
├── test_http_client.py      # Low-level client
├── test_client.py           # SDK client
└── test_filesystem.py       # Path handling
```

### Mock Strategy
```python
# Mock GeneratedClientWrapper
from unittest.mock import Mock, patch

@patch('model_download_sdk._http_client.GeneratedClientWrapper')
def test_download_models(mock_wrapper):
    mock_wrapper.return_value.call_download_models.return_value = Mock(
        job_ids=['job-1', 'job-2']
    )
    # Test SDK method
```

## Performance Considerations

### Job Polling
- Default poll interval: 2 seconds (configurable)
- Default max timeout: 3600 seconds (1 hour)
- Adapts to server response times

### Connection Management
- Lazy client initialization (created on first use)
- Proper cleanup with `close()`
- Context manager support for automatic cleanup

### Error Retry
- Currently: No automatic retry
- Future: Exponential backoff retry logic

## Limitations

### Current (Phase 2)
- No upload_model() - Returns NotImplementedError
- No streaming for large files
- No progress tracking
- No automatic retry

### Future (Phase 3+)
- Progress callbacks
- Streaming support
- Retry with exponential backoff
- Metrics collection

## Migration Guide

### From Phase 1 to Phase 2
No breaking changes! Phase 2 is backward compatible.

**What Changed**:
- All methods now actually work (not NotImplementedError)
- New logging throughout
- Better error messages
- Job polling implementation

**What Stayed Same**:
- SDK API (method signatures)
- Exception types
- Configuration options

## Troubleshooting

### Connection Refused
```
SDKConnectionError: Failed to connect to http://localhost:8200
```
- Ensure service is running
- Check base_url is correct
- Verify network connectivity

### Timeout During Download
```
SDKConnectionError: Job job-123 did not complete within 300s
```
- Increase timeout in config
- Check server logs for errors
- Consider smaller download

### Validation Error
```
ValidationError: Invalid hub: 'bad-hub'
```
- Check ModelHub enum values
- Verify model name and hub match
- See models.py for valid values

## Future Work

### Phase 3+
1. Unit test suite
2. Integration tests
3. upload_model() implementation
4. Progress callbacks
5. Streaming support
6. CLI improvements

### Performance
1. Connection pooling
2. Batch operations
3. Metrics collection
4. Retry logic

### Documentation
1. API reference
2. Architecture guide
3. Development guide
4. Troubleshooting guide

## References

- [Generated Client](generated/model-download-service-api-client/)
- [SDK Models](model_download_sdk/models.py)
- [Exception Hierarchy](model_download_sdk/exceptions.py)
- [Examples](examples.py)
- [Phase 2 Summary](PHASE_2_IMPLEMENTATION.md)
