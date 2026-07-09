# Model Download SDK Architecture

## Executive Summary

This document proposes a comprehensive abstraction layer for the Model Download Service OpenAPI client, designed to provide a Windows-compatible, user-friendly Python SDK that shields developers from generated client complexity while remaining maintainable through regenerations.

---

## 1. Available Endpoints Analysis

### 1.1 Health Management
```
GET /health
├─ Purpose: System health check
├─ Request: None (no parameters)
└─ Response: HealthResponse
    ├─ status: HealthResponseStatus (enum)
    └─ message: str
```

### 1.2 Model Operations (Core)
```
POST /models/download
├─ Purpose: Download and optionally convert models (async)
├─ Request: ModelDownloadRequest
│   ├─ models: list[ModelRequest]
│   │   ├─ name: str (model identifier, e.g., "microsoft/Phi-3.5-mini-instruct")
│   │   ├─ hub: ModelHub (enum: huggingface, ollama, geti, hls, openvino, ultralytics, pipeline-zoo-models)
│   │   ├─ type_: ModelType | Unset (enum: llm, embeddings, reranker, vision, vlm)
│   │   ├─ is_ovms: bool | Unset (OpenVINO conversion flag)
│   │   ├─ revision: str | Unset (specific version/branch)
│   │   └─ config: Config | Unset (OVMS conversion config)
│   └─ parallel_downloads: bool | Unset (currently unimplemented)
├─ Query Parameters:
│   └─ download_path: str (required)
└─ Response: DownloadResponse
    ├─ status: DownloadResponseStatus (enum)
    ├─ message: str
    └─ job_ids: list[str] (async task identifiers)
```

```
POST /models/upload
├─ Purpose: Upload custom models
├─ Request: UploadModelBody (multipart/form-data)
│   ├─ file: File (model file)
│   └─ [optional fields TBD]
└─ Response: UploadResponse
    ├─ status: (enum)
    ├─ message: str
    └─ job_id: str
```

```
GET /models/results
├─ Purpose: Retrieve completed model operations
├─ Request: None (no parameters)
└─ Response: ModelResultsResponse
    ├─ results: list[ModelResultsResponseResultsItem]
    │   ├─ model_name: str
    │   ├─ hub: str
    │   ├─ status: ModelResultStatus
    │   ├─ output_path: str
    │   └─ completion_time: datetime
    └─ [pagination fields if applicable]
```

### 1.3 Job Management (Async Task Tracking)
```
GET /jobs
├─ Purpose: List all jobs (downloads, conversions, uploads)
├─ Request: None (no parameters)
└─ Response: JobListResponse
    ├─ jobs: list[Job]
    │   ├─ job_id: str
    │   ├─ operation_type: JobOperationType (enum: DOWNLOAD, CONVERT, UPLOAD)
    │   ├─ model_name: str
    │   ├─ hub: str
    │   ├─ status: JobStatus (enum: pending, processing, completed, failed)
    │   ├─ output_dir: str
    │   ├─ plugin_name: str
    │   ├─ creation_time: datetime
    │   ├─ completion_time: datetime | None
    │   └─ error: str | None
    └─ [pagination fields if applicable]
```

```
GET /jobs/{job_id}
├─ Purpose: Get specific job details
├─ Path Parameters:
│   └─ job_id: str
├─ Request: None
└─ Response: Job (see above)
```

### 1.4 Plugins (System Capabilities)
```
GET /plugins
├─ Purpose: List available plugins (capabilities)
├─ Request: None
└─ Response: PluginsResponse
    ├─ available_plugins: list[PluginInfo]
    │   ├─ name: str
    │   ├─ version: str
    │   ├─ capabilities: PluginInfoCapabilities
    │   └─ [metadata]
    └─ [system info]
```

---

## 2. Request Models Deep Dive

### 2.1 Primary Request Models
| Model | Purpose | Required Fields | Optional Fields |
|-------|---------|-----------------|-----------------|
| `ModelDownloadRequest` | Batch download/convert | models[] | parallel_downloads |
| `ModelRequest` | Individual model spec | name, hub | type_, is_ovms, revision, config |
| `UploadModelBody` | File upload | file | metadata |

### 2.2 Enum Value Sets
```python
# ModelHub: 7 sources
HUGGINGFACE, OLLAMA, GETI, HLS, OPENVINO, ULTRALYTICS, PIPELINE_ZOO_MODELS

# ModelType: 5 types
LLM, EMBEDDINGS, RERANKER, VISION, VLM

# JobStatus: 4 states
PENDING, PROCESSING, COMPLETED, FAILED

# JobOperationType: 3 operations
DOWNLOAD, CONVERT, UPLOAD

# Response Statuses
DownloadResponseStatus: (values TBD from response models)
ModelResultStatus: (values TBD from response models)
```

---

## 3. Response Models Deep Dive

### 3.1 Primary Response Models
| Model | Contains | Error Variants |
|-------|----------|-----------------|
| `DownloadResponse` | job_ids[], status, message | 400, 422, 500 |
| `UploadResponse` | job_id, status, message | 400, 409, 413, 422, 500 |
| `Job` | Full task state + timestamps | None (used in lists) |
| `JobListResponse` | jobs[], pagination | None |
| `ModelResultsResponse` | results[], pagination | None |
| `HealthResponse` | status, message | None |
| `PluginsResponse` | available_plugins[] | None |

### 3.2 Error Responses
Generated client defines error response models:
- `DownloadModelsResponse400/422/500` (download errors)
- `UploadModelResponse400/409/413/422/500` (upload errors)
- `GetJobStatusResponse404` (not found)

---

## 4. Recommended SDK Abstraction Layer

### 4.1 Architecture Principles
1. **Generation Isolation**: Never import from generated package beyond client class
2. **Type Safety**: Use pydantic or attrs for SDK models (separate from generated)
3. **Async/Sync Parity**: Support both sync and async workflows
4. **Error Normalization**: Map generated errors to SDK-specific exceptions
5. **Windows Path Safety**: Normalize paths for Windows compatibility
6. **Lazy Client Initialization**: Defer connection until first API call

### 4.2 Proposed Structure

```
model_download_sdk/
├── __init__.py                 # Public API exports
├── exceptions.py               # SDK-specific exceptions
├── models/
│   ├── __init__.py
│   ├── request.py              # Request DTO layer (NOT imported from generated)
│   ├── response.py             # Response DTO layer (NOT imported from generated)
│   └── enums.py                # Re-exported enums from generated
├── client.py                   # Main sync client wrapper
├── async_client.py             # Async variant
├── job_tracker.py              # Job polling & state management
├── path_handler.py             # Windows path normalization
└── types.py                    # Type aliases & protocols
```

### 4.3 Core Client Classes

#### ModelDownloadSDK (Facade)
```python
class ModelDownloadSDK:
    """Main entry point for the SDK"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8200",
        timeout: float = 30.0,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
    ):
        """Initialize SDK with connection parameters"""
    
    # Health & System
    async def health_check(self) -> HealthStatus:
        """Check service health"""
    
    async def list_plugins(self) -> list[PluginCapability]:
        """Get available capabilities"""
    
    # Model Operations
    async def download_models(
        self,
        models: list[ModelSpec],
        output_directory: str,
        wait: bool = False,
        timeout: int | None = None,
    ) -> DownloadJob | list[Job]:
        """Download/convert models, optionally wait for completion"""
    
    async def upload_model(
        self,
        file_path: str,
        model_name: str,
        metadata: dict | None = None,
    ) -> UploadJob:
        """Upload custom model"""
    
    # Job Management
    async def get_job(self, job_id: str) -> Job:
        """Get job status"""
    
    async def list_jobs(
        self,
        status: JobStatus | None = None,
        operation: JobOperationType | None = None,
    ) -> list[Job]:
        """List jobs with optional filtering"""
    
    async def wait_for_job(
        self,
        job_id: str,
        timeout: int | None = None,
        poll_interval: float = 2.0,
    ) -> Job:
        """Poll until job completion"""
    
    # Model Results
    async def get_model_results(
        self,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ModelResult]:
        """Retrieve completed operations"""
    
    # Batch Operations
    async def download_and_wait(
        self,
        models: list[ModelSpec],
        output_directory: str,
        timeout: int | None = 3600,
    ) -> DownloadResult:
        """Convenience: download and wait for all jobs"""
```

#### ModelSpec (Request Builder)
```python
@dataclass
class ModelSpec:
    """User-friendly model specification"""
    
    name: str                              # e.g., "microsoft/Phi-3.5-mini-instruct"
    hub: ModelHub | str                    # Source
    type_: ModelType | str | None = None   # Optional: llm, embeddings, etc.
    convert_to_openvino: bool = False      # Conversion flag
    revision: str | None = None            # Branch/tag
    
    def to_generated_model(self) -> ModelRequest:
        """Convert to generated client model"""
        return ModelRequest(
            name=self.name,
            hub=ModelHub(self.hub) if isinstance(self.hub, str) else self.hub,
            type_=self._parse_type(),
            is_ovms=self.convert_to_openvino,
            revision=self.revision,
        )
```

#### Job (Result Wrapper)
```python
@dataclass
class Job:
    """SDK-level job representation"""
    
    id: str
    operation: JobOperationType
    model_name: str
    status: JobStatus
    created_at: datetime
    completed_at: datetime | None
    error: str | None
    output_directory: str | None
    plugin: str | None
    
    @property
    def is_complete(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)
    
    @property
    def is_success(self) -> bool:
        return self.status == JobStatus.COMPLETED
```

#### Exception Hierarchy
```python
class SDKError(Exception):
    """Base SDK exception"""
    pass

class ConnectionError(SDKError):
    """Cannot connect to service"""
    pass

class ValidationError(SDKError):
    """Invalid input parameters"""
    pass

class JobError(SDKError):
    """Job execution failed on server"""
    def __init__(self, job_id: str, message: str):
        self.job_id = job_id
        super().__init__(f"Job {job_id} failed: {message}")

class TimeoutError(SDKError):
    """Operation exceeded timeout"""
    pass

class NotFoundError(SDKError):
    """Resource not found"""
    pass
```

### 4.4 Usage Examples

#### Synchronous Wrapper
```python
# Synchronous convenience layer (thin wrapper around async)
class ModelDownloadClient:
    def __init__(self, base_url: str = "http://localhost:8200"):
        self._sdk = ModelDownloadSDK(base_url)
        self._loop = asyncio.new_event_loop()
    
    def download_model(
        self,
        model_name: str,
        hub: ModelHub,
        output_dir: str,
        convert_to_openvino: bool = False,
    ) -> str:
        """Download model (sync wrapper)"""
        spec = ModelSpec(
            name=model_name,
            hub=hub,
            convert_to_openvino=convert_to_openvino,
        )
        result = self._loop.run_until_complete(
            self._sdk.download_models([spec], output_dir)
        )
        return result.job_ids[0]
```

#### High-Level Usage
```python
# Create client
client = ModelDownloadSDK(base_url="http://localhost:8200")

# Download with auto-wait
models = [
    ModelSpec(name="microsoft/Phi-3.5", hub="huggingface", type_="llm"),
    ModelSpec(name="sentence-transformers/all-MiniLM", hub="huggingface", type_="embeddings"),
]

result = await client.download_and_wait(
    models=models,
    output_directory="~/models",
    timeout=3600,
)

for job in result.successful_jobs:
    print(f"✓ {job.model_name}: {job.output_directory}")

for job in result.failed_jobs:
    print(f"✗ {job.model_name}: {job.error}")
```

---

## 5. Potential Issues When Wrapping the Generated Client

### 5.1 Critical Issues

#### Issue 1: Circular Dependencies During Regeneration
**Problem**: SDK imports generated models directly. If generated package changes import structure during regeneration, SDK breaks.

**Solution**:
- Create SDK-specific models in `model_download_sdk/models/` that mirror generated models
- Use adapter layer to convert: `generated → SDK → user code`
- Never expose generated types in SDK public API

#### Issue 2: Breaking Changes in Generated Code
**Problem**: OpenAPI generator may change:
- Model field names (e.g., `type` → `type_`)
- Response structure
- Error handling patterns

**Solution**:
- Version-lock the generator and its OpenAPI spec
- Create `_generated_adapter.py` with translation functions
- Maintain compatibility layer even after regenerations

#### Issue 3: Windows Path Handling
**Problem**: Download paths may contain:
- UNC paths (`\\server\share`)
- Mixed separators
- Long paths (>260 chars on Windows)
- Special characters

**Solution**:
```python
# path_handler.py
def normalize_download_path(path: str, max_length: int = 260) -> str:
    """Normalize path for Windows compatibility"""
    normalized = Path(path).expanduser().resolve()
    
    if sys.platform == "win32":
        # Handle long paths on Windows 10+
        if len(str(normalized)) > max_length:
            normalized = Path(f"\\\\?\\{normalized}")
    
    return str(normalized)
```

#### Issue 4: Async/Sync Parity
**Problem**: Generated client supports both sync and async. SDK must provide both without duplicating business logic.

**Solution**:
- Implement core logic as async-first
- Use `asyncio.run()` for sync wrapper with proper event loop management
- Or use `anyio` for true async/sync agnostic code

#### Issue 5: Job State Management Race Conditions
**Problem**: Polling `/jobs/{job_id}` may return inconsistent state (e.g., `status=processing` but `completion_time` is set).

**Solution**:
```python
class JobTracker:
    def normalize_job_state(self, raw_job: GeneratedJob) -> Job:
        """Ensure job state is consistent"""
        # Force status based on completion_time
        if raw_job.completion_time and raw_job.status not in (COMPLETED, FAILED):
            raw_job.status = COMPLETED
        
        return Job(**raw_job.dict())
```

### 5.2 Operational Issues

#### Issue 6: Error Response Mapping
**Problem**: Generated client returns different error types for same logical error:
- `400` (bad request) - user input error
- `422` (validation) - invalid model spec
- `500` (server error) - service failure

**Solution**:
```python
class ErrorMapper:
    @staticmethod
    def to_sdk_error(response: Response) -> SDKError:
        """Normalize generated client errors to SDK exceptions"""
        status = response.status_code
        parsed = response.parsed
        
        if status == 400 or status == 422:
            return ValidationError(str(parsed))
        elif status == 404:
            return NotFoundError(str(parsed))
        elif status == 500:
            return ConnectionError(str(parsed))
        else:
            return SDKError(f"HTTP {status}: {parsed}")
```

#### Issue 7: Timeout Handling
**Problem**:
- httpx timeout != job timeout
- Download operations are async; httpx timeout is for API call only
- Need separate timeout for "wait for job completion"

**Solution**:
```python
class DownloadClient:
    async def download_and_wait(
        self,
        models: list[ModelSpec],
        output_dir: str,
        api_timeout: float = 30,      # httpx timeout
        job_timeout: int | None = 3600,  # poll timeout
    ) -> DownloadResult:
        # API call timeout
        response = await self._call_download_api(models, output_dir, timeout=api_timeout)
        
        # Job completion timeout
        return await self._wait_for_jobs(response.job_ids, timeout=job_timeout)
```

#### Issue 8: Multipart Upload Edge Cases
**Problem**: `UploadModelBody` uses multipart encoding. Issues include:
- Large file handling
- Streaming requirements
- Boundary encoding

**Solution**:
```python
async def upload_model(
    self,
    file_path: str,
    model_name: str,
    chunk_size: int = 8192,
) -> UploadJob:
    """Stream upload for large files"""
    
    file_size = Path(file_path).stat().st_size
    
    if file_size > 1024 * 1024:  # >1MB
        # Use streaming upload
        async with aiofiles.open(file_path, 'rb') as f:
            return await self._streaming_upload(f, model_name)
    else:
        # Use regular upload
        return await self._regular_upload(file_path, model_name)
```

### 5.3 Configuration Issues

#### Issue 9: SSL Certificate Validation
**Problem**: Windows may have:
- Corporate proxies with self-signed certs
- FIPS mode requirements
- Offline installations

**Solution**:
```python
class SDKConfig:
    verify_ssl: bool | str | ssl.SSLContext = True
    
    def __post_init__(self):
        if isinstance(self.verify_ssl, str):
            # Path to CA bundle
            self.verify_ssl = certifi.where()
```

#### Issue 10: Proxy Configuration
**Problem**: Windows users may be behind corporate proxies requiring authentication.

**Solution**:
```python
class SDKConfig:
    proxy_url: str | None = None
    proxy_auth: tuple[str, str] | None = None  # (user, password)
    
    def to_httpx_args(self) -> dict:
        args = {}
        if self.proxy_url:
            args['proxies'] = self.proxy_url
        return args
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] Analyze generated client structure
- [ ] Create SDK exception hierarchy
- [ ] Implement model/enum re-exports with type checking
- [ ] Create generated adapter layer
- [ ] Add Windows path normalization

### Phase 2: Core Client (Weeks 3-4)
- [ ] Implement `ModelDownloadSDK` base async client
- [ ] Add health check and plugin listing
- [ ] Implement job tracking and polling
- [ ] Add error mapping layer
- [ ] Create `ModelSpec` builder

### Phase 3: Sync & Convenience (Week 5)
- [ ] Create sync wrapper for async client
- [ ] Implement `download_and_wait()` convenience method
- [ ] Add upload support
- [ ] Create result filtering/pagination

### Phase 4: Polish & Windows (Week 6)
- [ ] Long path support on Windows
- [ ] Proxy/SSL configuration
- [ ] Comprehensive logging
- [ ] Usage documentation
- [ ] Integration tests

### Phase 5: Maintenance (Ongoing)
- [ ] Monitor generated client for breaking changes
- [ ] Update adapter layer on regeneration
- [ ] Version-lock OpenAPI generator version

---

## 7. Testing Strategy

### 7.1 Test Categories
1. **Unit Tests**: Models, adapters, error mapping
2. **Integration Tests**: Mock generated client calls
3. **E2E Tests**: Against actual service (if available)
4. **Windows-Specific Tests**: Path handling, long paths, UNC paths

### 7.2 Mock Layers
```python
# tests/mocks/generated_client.py
class MockGeneratedClient:
    """Mock generated client for testing"""
    
    async def download_models(self, request) -> Response:
        return Response(
            status_code=200,
            parsed=DownloadResponse(
                job_ids=["job-123"],
                status="success",
            ),
        )
```

---

## 8. Summary: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Async-first SDK | Matches OpenAPI client design; sync wrapper added for compatibility |
| Separate SDK models | Insulates SDK from generated code changes during regeneration |
| Adapter layer | Translates between generated and SDK models; centralized change management |
| Windows path handler | Addresses long paths (>260 chars), UNC paths, mixed separators |
| Job polling abstraction | Hides complexity of async task tracking; provides familiar blocking interface |
| Error normalization | Users see SDK exceptions, not generated client errors |
| Facade pattern | Reduces cognitive load; single entry point (`ModelDownloadSDK`) |

---

## Appendix: Generated Client Integration Points

### Safe to Use Directly (Stable)
- `client.Client` / `client.AuthenticatedClient` initialization
- Enum classes: `ModelHub`, `ModelType`, `JobStatus`
- Exception: `errors.UnexpectedStatus`

### Should Wrap (May Change)
- All model classes (use SDK equivalents instead)
- API functions (import via adapter only)
- Response parsing logic (abstract in adapter)

### Never Use Directly (Highly Unstable)
- `_get_kwargs()` functions
- `_parse_response()` functions
- `_build_response()` functions
- Internal types in `types.py`

---

## Glossary

| Term | Definition |
|------|-----------|
| **Job** | Async task representing download/conversion/upload operation |
| **Model Hub** | Model repository source (HuggingFace, Ollama, etc.) |
| **OpenVINO IR** | Intel's optimized inference model format |
| **OVMS** | OpenVINO Model Server |
| **Adapter Layer** | Translation code between generated and SDK models |
| **Sync Wrapper** | Async-to-sync converter for backward compatibility |
| **Path Normalization** | Converting paths to platform-safe format |

---

**Document Version**: 1.0  
**Date**: 2024-06-17  
**Status**: Proposed Architecture  
**Generated From**: Analysis of `generated/model-download-service-api-client/`
