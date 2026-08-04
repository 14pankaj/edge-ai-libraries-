from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..models.model_hub import ModelHub
from ..models.model_type import ModelType
from ..types import UNSET, Unset
from typing import cast

if TYPE_CHECKING:
  from ..models.config import Config





T = TypeVar("T", bound="ModelRequest")



@_attrs_define
class ModelRequest:
    """ 
        Attributes:
            name (str): The name/ID of the model (e.g., microsoft/Phi-3.5-mini-instruct) Example: microsoft/Phi-3.5-mini-
                instruct.
            hub (ModelHub): The model hub source to download from
            type_ (ModelType | Unset): The type of model (determines conversion behavior)
            is_ovms (bool | Unset): Whether to convert the model to OpenVINO IR format (requires OpenVINO plugin) Default:
                False.
            revision (str | Unset): Specific model revision/version to download Example: main.
            config (Config | Unset): Configuration for OVMS model conversion
     """

    name: str
    hub: ModelHub
    type_: ModelType | Unset = UNSET
    is_ovms: bool | Unset = False
    revision: str | Unset = UNSET
    config: Config | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.config import Config
        name = self.name

        hub = self.hub.value

        type_: str | Unset = UNSET
        if not isinstance(self.type_, Unset):
            type_ = self.type_.value


        is_ovms = self.is_ovms

        revision = self.revision

        config: dict[str, Any] | Unset = UNSET
        if not isinstance(self.config, Unset):
            config = self.config.to_dict()


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "name": name,
            "hub": hub,
        })
        if type_ is not UNSET:
            field_dict["type"] = type_
        if is_ovms is not UNSET:
            field_dict["is_ovms"] = is_ovms
        if revision is not UNSET:
            field_dict["revision"] = revision
        if config is not UNSET:
            field_dict["config"] = config

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config import Config
        d = dict(src_dict)
        name = d.pop("name")

        hub = ModelHub(d.pop("hub"))




        _type_ = d.pop("type", UNSET)
        type_: ModelType | Unset
        if isinstance(_type_,  Unset):
            type_ = UNSET
        else:
            type_ = ModelType(_type_)




        is_ovms = d.pop("is_ovms", UNSET)

        revision = d.pop("revision", UNSET)

        _config = d.pop("config", UNSET)
        config: Config | Unset
        if isinstance(_config,  Unset):
            config = UNSET
        else:
            config = Config.from_dict(_config)




        model_request = cls(
            name=name,
            hub=hub,
            type_=type_,
            is_ovms=is_ovms,
            revision=revision,
            config=config,
        )


        model_request.additional_properties = d
        return model_request

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
