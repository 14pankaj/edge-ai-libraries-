from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..models.job_operation_type import JobOperationType
from ..models.job_status import JobStatus
from ..types import UNSET, Unset
from typing import cast
import datetime






T = TypeVar("T", bound="Job")



@_attrs_define
class Job:
    """ 
        Attributes:
            job_id (str | Unset): Unique identifier for the job
            operation_type (JobOperationType | Unset): Type of operation
            model_name (str | Unset): Name of the model
            hub (str | Unset): Model hub source
            status (JobStatus | Unset): Status of a job
            output_dir (str | Unset): Output directory for the operation
            plugin_name (str | Unset): Plugin handling the operation
            creation_time (datetime.datetime | Unset): When the job was created
            completion_time (datetime.datetime | None | Unset): When the job was completed
            error (None | str | Unset): Error message if job failed
     """

    job_id: str | Unset = UNSET
    operation_type: JobOperationType | Unset = UNSET
    model_name: str | Unset = UNSET
    hub: str | Unset = UNSET
    status: JobStatus | Unset = UNSET
    output_dir: str | Unset = UNSET
    plugin_name: str | Unset = UNSET
    creation_time: datetime.datetime | Unset = UNSET
    completion_time: datetime.datetime | None | Unset = UNSET
    error: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        job_id = self.job_id

        operation_type: str | Unset = UNSET
        if not isinstance(self.operation_type, Unset):
            operation_type = self.operation_type.value


        model_name = self.model_name

        hub = self.hub

        status: str | Unset = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value


        output_dir = self.output_dir

        plugin_name = self.plugin_name

        creation_time: str | Unset = UNSET
        if not isinstance(self.creation_time, Unset):
            creation_time = self.creation_time.isoformat()

        completion_time: None | str | Unset
        if isinstance(self.completion_time, Unset):
            completion_time = UNSET
        elif isinstance(self.completion_time, datetime.datetime):
            completion_time = self.completion_time.isoformat()
        else:
            completion_time = self.completion_time

        error: None | str | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        else:
            error = self.error


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if operation_type is not UNSET:
            field_dict["operation_type"] = operation_type
        if model_name is not UNSET:
            field_dict["model_name"] = model_name
        if hub is not UNSET:
            field_dict["hub"] = hub
        if status is not UNSET:
            field_dict["status"] = status
        if output_dir is not UNSET:
            field_dict["output_dir"] = output_dir
        if plugin_name is not UNSET:
            field_dict["plugin_name"] = plugin_name
        if creation_time is not UNSET:
            field_dict["creation_time"] = creation_time
        if completion_time is not UNSET:
            field_dict["completion_time"] = completion_time
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        job_id = d.pop("job_id", UNSET)

        _operation_type = d.pop("operation_type", UNSET)
        operation_type: JobOperationType | Unset
        if isinstance(_operation_type,  Unset):
            operation_type = UNSET
        else:
            operation_type = JobOperationType(_operation_type)




        model_name = d.pop("model_name", UNSET)

        hub = d.pop("hub", UNSET)

        _status = d.pop("status", UNSET)
        status: JobStatus | Unset
        if isinstance(_status,  Unset):
            status = UNSET
        else:
            status = JobStatus(_status)




        output_dir = d.pop("output_dir", UNSET)

        plugin_name = d.pop("plugin_name", UNSET)

        _creation_time = d.pop("creation_time", UNSET)
        creation_time: datetime.datetime | Unset
        if isinstance(_creation_time,  Unset):
            creation_time = UNSET
        else:
            creation_time = datetime.datetime.fromisoformat(_creation_time)




        def _parse_completion_time(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                completion_time_type_0 = datetime.datetime.fromisoformat(data)



                return completion_time_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        completion_time = _parse_completion_time(d.pop("completion_time", UNSET))


        def _parse_error(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        error = _parse_error(d.pop("error", UNSET))


        job = cls(
            job_id=job_id,
            operation_type=operation_type,
            model_name=model_name,
            hub=hub,
            status=status,
            output_dir=output_dir,
            plugin_name=plugin_name,
            creation_time=creation_time,
            completion_time=completion_time,
            error=error,
        )


        job.additional_properties = d
        return job

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
