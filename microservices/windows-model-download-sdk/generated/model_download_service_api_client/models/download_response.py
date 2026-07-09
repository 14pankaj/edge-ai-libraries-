from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..models.download_response_status import DownloadResponseStatus
from ..types import UNSET, Unset
from typing import cast






T = TypeVar("T", bound="DownloadResponse")



@_attrs_define
class DownloadResponse:
    """ 
        Attributes:
            message (str | Unset): Status message Example: Started processing 1 model(s).
            job_ids (list[str] | Unset): List of job IDs created for the request
            status (DownloadResponseStatus | Unset): Overall status of the request
     """

    message: str | Unset = UNSET
    job_ids: list[str] | Unset = UNSET
    status: DownloadResponseStatus | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        message = self.message

        job_ids: list[str] | Unset = UNSET
        if not isinstance(self.job_ids, Unset):
            job_ids = self.job_ids



        status: str | Unset = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value



        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if message is not UNSET:
            field_dict["message"] = message
        if job_ids is not UNSET:
            field_dict["job_ids"] = job_ids
        if status is not UNSET:
            field_dict["status"] = status

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        message = d.pop("message", UNSET)

        job_ids = cast(list[str], d.pop("job_ids", UNSET))


        _status = d.pop("status", UNSET)
        status: DownloadResponseStatus | Unset
        if isinstance(_status,  Unset):
            status = UNSET
        else:
            status = DownloadResponseStatus(_status)




        download_response = cls(
            message=message,
            job_ids=job_ids,
            status=status,
        )


        download_response.additional_properties = d
        return download_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
