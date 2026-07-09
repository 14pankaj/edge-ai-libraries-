from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..models.device_type import DeviceType
from ..models.model_precision import ModelPrecision
from ..types import UNSET, Unset
from typing import cast






T = TypeVar("T", bound="Config")



@_attrs_define
class Config:
    """ Configuration for OVMS model conversion

        Attributes:
            precision (ModelPrecision | Unset): The precision format for model weights
            device (DeviceType | Unset): The target device type for model deployment
            cache_size (int | Unset): Cache size for model optimization
            quantize (None | str | Unset): Ultralytics quantization dataset used to enable INT8 export Example: coco128.
     """

    precision: ModelPrecision | Unset = UNSET
    device: DeviceType | Unset = UNSET
    cache_size: int | Unset = UNSET
    quantize: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        precision: str | Unset = UNSET
        if not isinstance(self.precision, Unset):
            precision = self.precision.value


        device: str | Unset = UNSET
        if not isinstance(self.device, Unset):
            device = self.device.value


        cache_size = self.cache_size

        quantize: None | str | Unset
        if isinstance(self.quantize, Unset):
            quantize = UNSET
        else:
            quantize = self.quantize


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if precision is not UNSET:
            field_dict["precision"] = precision
        if device is not UNSET:
            field_dict["device"] = device
        if cache_size is not UNSET:
            field_dict["cache_size"] = cache_size
        if quantize is not UNSET:
            field_dict["quantize"] = quantize

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        _precision = d.pop("precision", UNSET)
        precision: ModelPrecision | Unset
        if isinstance(_precision,  Unset):
            precision = UNSET
        else:
            precision = ModelPrecision(_precision)




        _device = d.pop("device", UNSET)
        device: DeviceType | Unset
        if isinstance(_device,  Unset):
            device = UNSET
        else:
            device = DeviceType(_device)




        cache_size = d.pop("cache_size", UNSET)

        def _parse_quantize(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        quantize = _parse_quantize(d.pop("quantize", UNSET))


        config = cls(
            precision=precision,
            device=device,
            cache_size=cache_size,
            quantize=quantize,
        )


        config.additional_properties = d
        return config

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
