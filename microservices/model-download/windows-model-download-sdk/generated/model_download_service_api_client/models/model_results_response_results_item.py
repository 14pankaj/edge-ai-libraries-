from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast
import datetime






T = TypeVar("T", bound="ModelResultsResponseResultsItem")



@_attrs_define
class ModelResultsResponseResultsItem:
    """ 
        Attributes:
            job_id (str | Unset):
            model_name (str | Unset):
            hub (str | Unset):
            operation_type (str | Unset):
            status (str | Unset):
            model_path (str | Unset):
            is_ovms (bool | Unset):
            completion_time (datetime.datetime | Unset):
     """

    job_id: str | Unset = UNSET
    model_name: str | Unset = UNSET
    hub: str | Unset = UNSET
    operation_type: str | Unset = UNSET
    status: str | Unset = UNSET
    model_path: str | Unset = UNSET
    is_ovms: bool | Unset = UNSET
    completion_time: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        job_id = self.job_id

        model_name = self.model_name

        hub = self.hub

        operation_type = self.operation_type

        status = self.status

        model_path = self.model_path

        is_ovms = self.is_ovms

        completion_time: str | Unset = UNSET
        if not isinstance(self.completion_time, Unset):
            completion_time = self.completion_time.isoformat()


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if model_name is not UNSET:
            field_dict["model_name"] = model_name
        if hub is not UNSET:
            field_dict["hub"] = hub
        if operation_type is not UNSET:
            field_dict["operation_type"] = operation_type
        if status is not UNSET:
            field_dict["status"] = status
        if model_path is not UNSET:
            field_dict["model_path"] = model_path
        if is_ovms is not UNSET:
            field_dict["is_ovms"] = is_ovms
        if completion_time is not UNSET:
            field_dict["completion_time"] = completion_time

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        job_id = d.pop("job_id", UNSET)

        model_name = d.pop("model_name", UNSET)

        hub = d.pop("hub", UNSET)

        operation_type = d.pop("operation_type", UNSET)

        status = d.pop("status", UNSET)

        model_path = d.pop("model_path", UNSET)

        is_ovms = d.pop("is_ovms", UNSET)

        _completion_time = d.pop("completion_time", UNSET)
        completion_time: datetime.datetime | Unset
        if isinstance(_completion_time,  Unset):
            completion_time = UNSET
        else:
            completion_time = datetime.datetime.fromisoformat(_completion_time)




        model_results_response_results_item = cls(
            job_id=job_id,
            model_name=model_name,
            hub=hub,
            operation_type=operation_type,
            status=status,
            model_path=model_path,
            is_ovms=is_ovms,
            completion_time=completion_time,
        )


        model_results_response_results_item.additional_properties = d
        return model_results_response_results_item

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
