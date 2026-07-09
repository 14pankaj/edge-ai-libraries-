from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..models.model_result_status import ModelResultStatus
from ..types import UNSET, Unset
from typing import cast






T = TypeVar("T", bound="ModelResult")



@_attrs_define
class ModelResult:
    """ 
        Attributes:
            status (ModelResultStatus): The status of the model download/conversion
            model_name (str): The name of the model
            model_path (None | str | Unset): Path where the model was downloaded/converted
            error (None | str | Unset): Error message if status is error
            is_ovms (bool | None | Unset): Whether the model was converted to OVMS format
     """

    status: ModelResultStatus
    model_name: str
    model_path: None | str | Unset = UNSET
    error: None | str | Unset = UNSET
    is_ovms: bool | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        status = self.status.value

        model_name = self.model_name

        model_path: None | str | Unset
        if isinstance(self.model_path, Unset):
            model_path = UNSET
        else:
            model_path = self.model_path

        error: None | str | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        else:
            error = self.error

        is_ovms: bool | None | Unset
        if isinstance(self.is_ovms, Unset):
            is_ovms = UNSET
        else:
            is_ovms = self.is_ovms


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "status": status,
            "model_name": model_name,
        })
        if model_path is not UNSET:
            field_dict["model_path"] = model_path
        if error is not UNSET:
            field_dict["error"] = error
        if is_ovms is not UNSET:
            field_dict["is_ovms"] = is_ovms

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        status = ModelResultStatus(d.pop("status"))




        model_name = d.pop("model_name")

        def _parse_model_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        model_path = _parse_model_path(d.pop("model_path", UNSET))


        def _parse_error(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        error = _parse_error(d.pop("error", UNSET))


        def _parse_is_ovms(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        is_ovms = _parse_is_ovms(d.pop("is_ovms", UNSET))


        model_result = cls(
            status=status,
            model_name=model_name,
            model_path=model_path,
            error=error,
            is_ovms=is_ovms,
        )


        model_result.additional_properties = d
        return model_result

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
