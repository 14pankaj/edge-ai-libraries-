from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast

if TYPE_CHECKING:
  from ..models.upload_model_response_422_detail_item import UploadModelResponse422DetailItem





T = TypeVar("T", bound="UploadModelResponse422")



@_attrs_define
class UploadModelResponse422:
    """ 
        Attributes:
            detail (list[UploadModelResponse422DetailItem] | Unset):
     """

    detail: list[UploadModelResponse422DetailItem] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.upload_model_response_422_detail_item import UploadModelResponse422DetailItem
        detail: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.detail, Unset):
            detail = []
            for detail_item_data in self.detail:
                detail_item = detail_item_data.to_dict()
                detail.append(detail_item)




        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if detail is not UNSET:
            field_dict["detail"] = detail

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.upload_model_response_422_detail_item import UploadModelResponse422DetailItem
        d = dict(src_dict)
        _detail = d.pop("detail", UNSET)
        detail: list[UploadModelResponse422DetailItem] | Unset = UNSET
        if _detail is not UNSET:
            detail = []
            for detail_item_data in _detail:
                detail_item = UploadModelResponse422DetailItem.from_dict(detail_item_data)



                detail.append(detail_item)


        upload_model_response_422 = cls(
            detail=detail,
        )


        upload_model_response_422.additional_properties = d
        return upload_model_response_422

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
