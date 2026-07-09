from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast

if TYPE_CHECKING:
  from ..models.plugins_response_available_plugins import PluginsResponseAvailablePlugins





T = TypeVar("T", bound="PluginsResponse")



@_attrs_define
class PluginsResponse:
    """ 
        Attributes:
            available_plugins (PluginsResponseAvailablePlugins | Unset):
            total_count (int | Unset): Total number of plugins
            available_count (int | Unset): Number of available plugins
            activation_instructions (str | Unset): Instructions for enabling/disabling plugins
     """

    available_plugins: PluginsResponseAvailablePlugins | Unset = UNSET
    total_count: int | Unset = UNSET
    available_count: int | Unset = UNSET
    activation_instructions: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.plugins_response_available_plugins import PluginsResponseAvailablePlugins
        available_plugins: dict[str, Any] | Unset = UNSET
        if not isinstance(self.available_plugins, Unset):
            available_plugins = self.available_plugins.to_dict()

        total_count = self.total_count

        available_count = self.available_count

        activation_instructions = self.activation_instructions


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
        if available_plugins is not UNSET:
            field_dict["available_plugins"] = available_plugins
        if total_count is not UNSET:
            field_dict["total_count"] = total_count
        if available_count is not UNSET:
            field_dict["available_count"] = available_count
        if activation_instructions is not UNSET:
            field_dict["activation_instructions"] = activation_instructions

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.plugins_response_available_plugins import PluginsResponseAvailablePlugins
        d = dict(src_dict)
        _available_plugins = d.pop("available_plugins", UNSET)
        available_plugins: PluginsResponseAvailablePlugins | Unset
        if isinstance(_available_plugins,  Unset):
            available_plugins = UNSET
        else:
            available_plugins = PluginsResponseAvailablePlugins.from_dict(_available_plugins)




        total_count = d.pop("total_count", UNSET)

        available_count = d.pop("available_count", UNSET)

        activation_instructions = d.pop("activation_instructions", UNSET)

        plugins_response = cls(
            available_plugins=available_plugins,
            total_count=total_count,
            available_count=available_count,
            activation_instructions=activation_instructions,
        )


        plugins_response.additional_properties = d
        return plugins_response

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
