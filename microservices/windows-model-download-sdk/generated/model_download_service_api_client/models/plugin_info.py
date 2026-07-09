from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast

if TYPE_CHECKING:
  from ..models.plugin_info_capabilities import PluginInfoCapabilities





T = TypeVar("T", bound="PluginInfo")



@_attrs_define
class PluginInfo:
    """ 
        Attributes:
            name (str | Unset): Plugin name
            type_ (str | Unset): Plugin type (hub or conversion)
            description (str | Unset): Plugin description
            capabilities (PluginInfoCapabilities | Unset):
            available (bool | Unset): Whether the plugin is available
            unavailable_reason (None | str | Unset): Reason if plugin is not available
     """

    name: str | Unset = UNSET
    type_: str | Unset = UNSET
    description: str | Unset = UNSET
    capabilities: PluginInfoCapabilities | Unset = UNSET
    available: bool | Unset = UNSET
    unavailable_reason: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.plugin_info_capabilities import PluginInfoCapabilities
        name = self.name

        type_ = self.type_

        description = self.description

        capabilities: dict[str, Any] | Unset = UNSET
        if not isinstance(self.capabilities, Unset):
            capabilities = self.capabilities.to_dict()

        available = self.available

        unavailable_reason: None | str | Unset
        if isinstance(self.unavailable_reason, Unset):
            unavailable_reason = UNSET
        else:
            unavailable_reason = self.unavailable_reason


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if name is not UNSET:
            field_dict["name"] = name
        if type_ is not UNSET:
            field_dict["type"] = type_
        if description is not UNSET:
            field_dict["description"] = description
        if capabilities is not UNSET:
            field_dict["capabilities"] = capabilities
        if available is not UNSET:
            field_dict["available"] = available
        if unavailable_reason is not UNSET:
            field_dict["unavailable_reason"] = unavailable_reason

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.plugin_info_capabilities import PluginInfoCapabilities
        d = dict(src_dict)
        name = d.pop("name", UNSET)

        type_ = d.pop("type", UNSET)

        description = d.pop("description", UNSET)

        _capabilities = d.pop("capabilities", UNSET)
        capabilities: PluginInfoCapabilities | Unset
        if isinstance(_capabilities,  Unset):
            capabilities = UNSET
        else:
            capabilities = PluginInfoCapabilities.from_dict(_capabilities)




        available = d.pop("available", UNSET)

        def _parse_unavailable_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        unavailable_reason = _parse_unavailable_reason(d.pop("unavailable_reason", UNSET))


        plugin_info = cls(
            name=name,
            type_=type_,
            description=description,
            capabilities=capabilities,
            available=available,
            unavailable_reason=unavailable_reason,
        )


        plugin_info.additional_properties = d
        return plugin_info

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
