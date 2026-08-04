from enum import Enum

class DeviceType(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"

    def __str__(self) -> str:
        return str(self.value)
