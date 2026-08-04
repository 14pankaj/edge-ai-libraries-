from enum import Enum

class JobOperationType(str, Enum):
    CONVERT = "convert"
    DOWNLOAD = "download"

    def __str__(self) -> str:
        return str(self.value)
