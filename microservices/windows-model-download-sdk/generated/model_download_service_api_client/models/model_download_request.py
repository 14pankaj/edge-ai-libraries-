from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast

if TYPE_CHECKING:
  from ..models.model_request import ModelRequest





T = TypeVar("T", bound="ModelDownloadRequest")



@_attrs_define
class ModelDownloadRequest:
    """ 
        Attributes:
            models (list[ModelRequest]): List of models to download and/or convert
            parallel_downloads (bool | Unset): Whether to download models in parallel (currently not implemented) Default:
                False.
     """

    models: list[ModelRequest]
    parallel_downloads: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.model_request import ModelRequest
        models = []
        for models_item_data in self.models:
            models_item = models_item_data.to_dict()
            models.append(models_item)



        parallel_downloads = self.parallel_downloads


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "models": models,
        })
        if parallel_downloads is not UNSET:
            field_dict["parallel_downloads"] = parallel_downloads

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_request import ModelRequest
        d = dict(src_dict)
        models = []
        _models = d.pop("models")
        for models_item_data in (_models):
            models_item = ModelRequest.from_dict(models_item_data)



            models.append(models_item)


        parallel_downloads = d.pop("parallel_downloads", UNSET)

        model_download_request = cls(
            models=models,
            parallel_downloads=parallel_downloads,
        )


        model_download_request.additional_properties = d
        return model_download_request

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
