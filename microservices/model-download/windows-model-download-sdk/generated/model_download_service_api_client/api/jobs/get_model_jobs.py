from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ...client import AuthenticatedClient, Client
from ...types import Response, UNSET
from ... import errors

from ...models.get_model_jobs_response_404 import GetModelJobsResponse404
from ...models.job_list_response import JobListResponse
from typing import cast



def _get_kwargs(
    *,
    model_name: str,

) -> dict[str, Any]:
    

    

    params: dict[str, Any] = {}

    params["model_name"] = model_name


    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}


    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/models/jobs",
        "params": params,
    }


    return _kwargs



def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> GetModelJobsResponse404 | JobListResponse | None:
    if response.status_code == 200:
        response_200 = JobListResponse.from_dict(response.json())



        return response_200

    if response.status_code == 404:
        response_404 = GetModelJobsResponse404.from_dict(response.json())



        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[GetModelJobsResponse404 | JobListResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    model_name: str,

) -> Response[GetModelJobsResponse404 | JobListResponse]:
    """ Get jobs for a specific model

     Retrieve all jobs (downloads and conversions) related to a specific model

    Args:
        model_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetModelJobsResponse404 | JobListResponse]
     """


    kwargs = _get_kwargs(
        model_name=model_name,

    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)

def sync(
    *,
    client: AuthenticatedClient | Client,
    model_name: str,

) -> GetModelJobsResponse404 | JobListResponse | None:
    """ Get jobs for a specific model

     Retrieve all jobs (downloads and conversions) related to a specific model

    Args:
        model_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetModelJobsResponse404 | JobListResponse
     """


    return sync_detailed(
        client=client,
model_name=model_name,

    ).parsed

async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    model_name: str,

) -> Response[GetModelJobsResponse404 | JobListResponse]:
    """ Get jobs for a specific model

     Retrieve all jobs (downloads and conversions) related to a specific model

    Args:
        model_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetModelJobsResponse404 | JobListResponse]
     """


    kwargs = _get_kwargs(
        model_name=model_name,

    )

    response = await client.get_async_httpx_client().request(
        **kwargs
    )

    return _build_response(client=client, response=response)

async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    model_name: str,

) -> GetModelJobsResponse404 | JobListResponse | None:
    """ Get jobs for a specific model

     Retrieve all jobs (downloads and conversions) related to a specific model

    Args:
        model_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetModelJobsResponse404 | JobListResponse
     """


    return (await asyncio_detailed(
        client=client,
model_name=model_name,

    )).parsed
