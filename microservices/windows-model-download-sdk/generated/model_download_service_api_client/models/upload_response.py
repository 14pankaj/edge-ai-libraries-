from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset






T = TypeVar("T", bound="UploadResponse")



@_attrs_define
class UploadResponse:
    """ 
        Attributes:
            status (str | Unset):  Example: success.
            message (str | Unset):  Example: Model 'my_custom_model' uploaded successfully..
            job_id (str | Unset): Job ID for the completed upload operation
            model_name (str | Unset): Sanitized model name used for storage
            model_path (str | Unset): Final extracted model path
     """

    status: str | Unset = UNSET
    message: str | Unset = UNSET
    job_id: str | Unset = UNSET
    model_name: str | Unset = UNSET
    model_path: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        status = self.status

        message = self.message

        job_id = self.job_id

        model_name = self.model_name

        model_path = self.model_path


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if status is not UNSET:
            field_dict["status"] = status
        if message is not UNSET:
            field_dict["message"] = message
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if model_name is not UNSET:
            field_dict["model_name"] = model_name
        if model_path is not UNSET:
            field_dict["model_path"] = model_path

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        status = d.pop("status", UNSET)

        message = d.pop("message", UNSET)

        job_id = d.pop("job_id", UNSET)

        model_name = d.pop("model_name", UNSET)

        model_path = d.pop("model_path", UNSET)

        upload_response = cls(
            status=status,
            message=message,
            job_id=job_id,
            model_name=model_name,
            model_path=model_path,
        )


        upload_response.additional_properties = d
        return upload_response

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
