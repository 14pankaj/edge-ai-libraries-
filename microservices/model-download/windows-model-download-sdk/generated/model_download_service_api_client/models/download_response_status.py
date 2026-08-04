from enum import Enum

class DownloadResponseStatus(str, Enum):
    PROCESSING = "processing"

    def __str__(self) -> str:
        return str(self.value)
