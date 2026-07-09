from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ...client import AuthenticatedClient, Client
from ...types import Response, UNSET
from ... import errors

from ...models.upload_model_body import UploadModelBody
from ...models.upload_model_response_400 import UploadModelResponse400
from ...models.upload_model_response_409 import UploadModelResponse409
from ...models.upload_model_response_413 import UploadModelResponse413
from ...models.upload_model_response_422 import UploadModelResponse422
from ...models.upload_model_response_500 import UploadModelResponse500
from ...models.upload_response import UploadResponse
from typing import cast



def _get_kwargs(
    *,
    body: UploadModelBody,

) -> dict[str, Any]:
    headers: dict[str, Any] = {}


    

    

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/models/upload",
    }

    _kwargs["files"] = body.to_multipart()

    headers["Content-Type"] = "multipart/form-data; boundary=+++"

    _kwargs["headers"] = headers
    return _kwargs



def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse | None:
    if response.status_code == 200:
        response_200 = UploadResponse.from_dict(response.json())



        return response_200

    if response.status_code == 400:
        response_400 = UploadModelResponse400.from_dict(response.json())



        return response_400

    if response.status_code == 409:
        response_409 = UploadModelResponse409.from_dict(response.json())



        return response_409

    if response.status_code == 413:
        response_413 = UploadModelResponse413.from_dict(response.json())



        return response_413

    if response.status_code == 422:
        response_422 = UploadModelResponse422.from_dict(response.json())



        return response_422

    if response.status_code == 500:
        response_500 = UploadModelResponse500.from_dict(response.json())



        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: UploadModelBody,

) -> Response[UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse]:
    """ Upload Custom model ZIP File

     Upload a ZIP file that contains `model.xml` and `model.bin` at the ZIP root.
    The model is extracted under:
    `/opt/models/custom_uploaded_models/{provider}/{framework}/{model_name}/[{precision}/]`

    **Validation and behavior:**
    - `model_name` is required and sanitized to lowercase with safe characters
    - File size limit is enforced (default 500MB via `MAX_UPLOAD_SIZE_MB`)
    - Files are read in chunks (default 8KB via `UPLOAD_CHUNK_SIZE_KB`) to prevent memory exhaustion
    with large uploads
    - If target model path already exists, the API returns `409`
    - Successful upload is registered as a completed job and appears in `/models/results`

    Args:
        body (UploadModelBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse]
     """


    kwargs = _get_kwargs(
        body=body,

    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)

def sync(
    *,
    client: AuthenticatedClient | Client,
    body: UploadModelBody,

) -> UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse | None:
    """ Upload Custom model ZIP File

     Upload a ZIP file that contains `model.xml` and `model.bin` at the ZIP root.
    The model is extracted under:
    `/opt/models/custom_uploaded_models/{provider}/{framework}/{model_name}/[{precision}/]`

    **Validation and behavior:**
    - `model_name` is required and sanitized to lowercase with safe characters
    - File size limit is enforced (default 500MB via `MAX_UPLOAD_SIZE_MB`)
    - Files are read in chunks (default 8KB via `UPLOAD_CHUNK_SIZE_KB`) to prevent memory exhaustion
    with large uploads
    - If target model path already exists, the API returns `409`
    - Successful upload is registered as a completed job and appears in `/models/results`

    Args:
        body (UploadModelBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse
     """


    return sync_detailed(
        client=client,
body=body,

    ).parsed

async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: UploadModelBody,

) -> Response[UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse]:
    """ Upload Custom model ZIP File

     Upload a ZIP file that contains `model.xml` and `model.bin` at the ZIP root.
    The model is extracted under:
    `/opt/models/custom_uploaded_models/{provider}/{framework}/{model_name}/[{precision}/]`

    **Validation and behavior:**
    - `model_name` is required and sanitized to lowercase with safe characters
    - File size limit is enforced (default 500MB via `MAX_UPLOAD_SIZE_MB`)
    - Files are read in chunks (default 8KB via `UPLOAD_CHUNK_SIZE_KB`) to prevent memory exhaustion
    with large uploads
    - If target model path already exists, the API returns `409`
    - Successful upload is registered as a completed job and appears in `/models/results`

    Args:
        body (UploadModelBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse]
     """


    kwargs = _get_kwargs(
        body=body,

    )

    response = await client.get_async_httpx_client().request(
        **kwargs
    )

    return _build_response(client=client, response=response)

async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: UploadModelBody,

) -> UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse | None:
    """ Upload Custom model ZIP File

     Upload a ZIP file that contains `model.xml` and `model.bin` at the ZIP root.
    The model is extracted under:
    `/opt/models/custom_uploaded_models/{provider}/{framework}/{model_name}/[{precision}/]`

    **Validation and behavior:**
    - `model_name` is required and sanitized to lowercase with safe characters
    - File size limit is enforced (default 500MB via `MAX_UPLOAD_SIZE_MB`)
    - Files are read in chunks (default 8KB via `UPLOAD_CHUNK_SIZE_KB`) to prevent memory exhaustion
    with large uploads
    - If target model path already exists, the API returns `409`
    - Successful upload is registered as a completed job and appears in `/models/results`

    Args:
        body (UploadModelBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        UploadModelResponse400 | UploadModelResponse409 | UploadModelResponse413 | UploadModelResponse422 | UploadModelResponse500 | UploadResponse
     """


    return (await asyncio_detailed(
        client=client,
body=body,

    )).parsed
