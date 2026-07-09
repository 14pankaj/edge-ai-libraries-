from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field
import json
from .. import types

from ..types import UNSET, Unset

from ..types import File, FileTypes
from ..types import UNSET, Unset
from io import BytesIO
from typing import cast






T = TypeVar("T", bound="UploadModelBody")



@_attrs_define
class UploadModelBody:
    """ 
        Attributes:
            file (File): ZIP file containing `model.xml` and `model.bin`
            model_name (str): Model name provided by user Example: my_custom_model.
            provider (str | Unset): Provider segment in target path Example: geti.
            framework (str | Unset): Framework segment in target path Example: openvino.
            precision (None | str | Unset): Optional precision folder (for example FP16, FP32, or INT8) Example: FP16.
     """

    file: File
    model_name: str
    provider: str | Unset = UNSET
    framework: str | Unset = UNSET
    precision: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        file = self.file.to_tuple()


        model_name = self.model_name

        provider = self.provider

        framework = self.framework

        precision: None | str | Unset
        if isinstance(self.precision, Unset):
            precision = UNSET
        else:
            precision = self.precision


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "file": file,
            "model_name": model_name,
        })
        if provider is not UNSET:
            field_dict["provider"] = provider
        if framework is not UNSET:
            field_dict["framework"] = framework
        if precision is not UNSET:
            field_dict["precision"] = precision

        return field_dict


    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        files.append(("file", self.file.to_tuple()))



        files.append(("model_name", (None, str(self.model_name).encode(), "text/plain")))



        if not isinstance(self.provider, Unset):
            files.append(("provider", (None, str(self.provider).encode(), "text/plain")))



        if not isinstance(self.framework, Unset):
            files.append(("framework", (None, str(self.framework).encode(), "text/plain")))



        if not isinstance(self.precision, Unset):
            if isinstance(self.precision, str):

                files.append(("precision", (None, str(self.precision).encode(), "text/plain")))
            else:
                files.append(("precision", (None, str(self.precision).encode(), "text/plain")))



        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))



        return files


    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        file = File(
             payload = BytesIO(d.pop("file"))
        )




        model_name = d.pop("model_name")

        provider = d.pop("provider", UNSET)

        framework = d.pop("framework", UNSET)

        def _parse_precision(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        precision = _parse_precision(d.pop("precision", UNSET))


        upload_model_body = cls(
            file=file,
            model_name=model_name,
            provider=provider,
            framework=framework,
            precision=precision,
        )


        upload_model_body.additional_properties = d
        return upload_model_body

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
