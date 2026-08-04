from enum import Enum

class ModelPrecision(str, Enum):
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"

    def __str__(self) -> str:
        return str(self.value)
