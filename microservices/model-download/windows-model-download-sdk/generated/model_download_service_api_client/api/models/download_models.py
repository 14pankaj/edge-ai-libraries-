from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ...client import AuthenticatedClient, Client
from ...types import Response, UNSET
from ... import errors

from ...models.download_models_response_400 import DownloadModelsResponse400
from ...models.download_models_response_422 import DownloadModelsResponse422
from ...models.download_models_response_500 import DownloadModelsResponse500
from ...models.download_response import DownloadResponse
from ...models.model_download_request import ModelDownloadRequest
from typing import cast



def _get_kwargs(
    *,
    body: ModelDownloadRequest,
    download_path: str,

) -> dict[str, Any]:
    headers: dict[str, Any] = {}


    

    params: dict[str, Any] = {}

    params["download_path"] = download_path


    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}


    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/models/download",
        "params": params,
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs



def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse | None:
    if response.status_code == 200:
        response_200 = DownloadResponse.from_dict(response.json())



        return response_200

    if response.status_code == 400:
        response_400 = DownloadModelsResponse400.from_dict(response.json())



        return response_400

    if response.status_code == 422:
        response_422 = DownloadModelsResponse422.from_dict(response.json())



        return response_422

    if response.status_code == 500:
        response_500 = DownloadModelsResponse500.from_dict(response.json())



        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ModelDownloadRequest,
    download_path: str,

) -> Response[DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse]:
    """ Download and optionally convert models

     Downloads one or more models from HuggingFace, Ollama, Ultralytics, Pipeline Zoo Models, hls or
    Geti™ and optionally converts them
    to OpenVINO IR format for huggingface hub models. This endpoint processes requests asynchronously
    and returns job IDs immediately.

    **Conversion Behavior:**
    Models will be converted to OpenVINO format if:
    1. `is_ovms` is set to `true` in the request
    2. `type` can be set to 'llm,embeddings,reranker, vlm or vision' in the request

    **Authentication:**
    - HUGGINGFACEHUB_API_TOKEN is optional for public HuggingFace models
    - HUGGINGFACEHUB_API_TOKEN is required for gated/private HuggingFace models and for conversion
    - No authentication needed for Ollama, Ultralytics, Pipeline Zoo Models, Geti™, or HLS models

    **Job Processing:**
    - Each model download creates a separate job
    - If conversion is requested, an additional conversion job is created
    - Use the returned job IDs to track progress via `/jobs/{job_id}` endpoint

    Args:
        download_path (str):
        body (ModelDownloadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse]
     """


    kwargs = _get_kwargs(
        body=body,
download_path=download_path,

    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)

def sync(
    *,
    client: AuthenticatedClient | Client,
    body: ModelDownloadRequest,
    download_path: str,

) -> DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse | None:
    """ Download and optionally convert models

     Downloads one or more models from HuggingFace, Ollama, Ultralytics, Pipeline Zoo Models, hls or
    Geti™ and optionally converts them
    to OpenVINO IR format for huggingface hub models. This endpoint processes requests asynchronously
    and returns job IDs immediately.

    **Conversion Behavior:**
    Models will be converted to OpenVINO format if:
    1. `is_ovms` is set to `true` in the request
    2. `type` can be set to 'llm,embeddings,reranker, vlm or vision' in the request

    **Authentication:**
    - HUGGINGFACEHUB_API_TOKEN is optional for public HuggingFace models
    - HUGGINGFACEHUB_API_TOKEN is required for gated/private HuggingFace models and for conversion
    - No authentication needed for Ollama, Ultralytics, Pipeline Zoo Models, Geti™, or HLS models

    **Job Processing:**
    - Each model download creates a separate job
    - If conversion is requested, an additional conversion job is created
    - Use the returned job IDs to track progress via `/jobs/{job_id}` endpoint

    Args:
        download_path (str):
        body (ModelDownloadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse
     """


    return sync_detailed(
        client=client,
body=body,
download_path=download_path,

    ).parsed

async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ModelDownloadRequest,
    download_path: str,

) -> Response[DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse]:
    """ Download and optionally convert models

     Downloads one or more models from HuggingFace, Ollama, Ultralytics, Pipeline Zoo Models, hls or
    Geti™ and optionally converts them
    to OpenVINO IR format for huggingface hub models. This endpoint processes requests asynchronously
    and returns job IDs immediately.

    **Conversion Behavior:**
    Models will be converted to OpenVINO format if:
    1. `is_ovms` is set to `true` in the request
    2. `type` can be set to 'llm,embeddings,reranker, vlm or vision' in the request

    **Authentication:**
    - HUGGINGFACEHUB_API_TOKEN is optional for public HuggingFace models
    - HUGGINGFACEHUB_API_TOKEN is required for gated/private HuggingFace models and for conversion
    - No authentication needed for Ollama, Ultralytics, Pipeline Zoo Models, Geti™, or HLS models

    **Job Processing:**
    - Each model download creates a separate job
    - If conversion is requested, an additional conversion job is created
    - Use the returned job IDs to track progress via `/jobs/{job_id}` endpoint

    Args:
        download_path (str):
        body (ModelDownloadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse]
     """


    kwargs = _get_kwargs(
        body=body,
download_path=download_path,

    )

    response = await client.get_async_httpx_client().request(
        **kwargs
    )

    return _build_response(client=client, response=response)

async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ModelDownloadRequest,
    download_path: str,

) -> DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse | None:
    """ Download and optionally convert models

     Downloads one or more models from HuggingFace, Ollama, Ultralytics, Pipeline Zoo Models, hls or
    Geti™ and optionally converts them
    to OpenVINO IR format for huggingface hub models. This endpoint processes requests asynchronously
    and returns job IDs immediately.

    **Conversion Behavior:**
    Models will be converted to OpenVINO format if:
    1. `is_ovms` is set to `true` in the request
    2. `type` can be set to 'llm,embeddings,reranker, vlm or vision' in the request

    **Authentication:**
    - HUGGINGFACEHUB_API_TOKEN is optional for public HuggingFace models
    - HUGGINGFACEHUB_API_TOKEN is required for gated/private HuggingFace models and for conversion
    - No authentication needed for Ollama, Ultralytics, Pipeline Zoo Models, Geti™, or HLS models

    **Job Processing:**
    - Each model download creates a separate job
    - If conversion is requested, an additional conversion job is created
    - Use the returned job IDs to track progress via `/jobs/{job_id}` endpoint

    Args:
        download_path (str):
        body (ModelDownloadRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DownloadModelsResponse400 | DownloadModelsResponse422 | DownloadModelsResponse500 | DownloadResponse
     """


    return (await asyncio_detailed(
        client=client,
body=body,
download_path=download_path,

    )).parsed
