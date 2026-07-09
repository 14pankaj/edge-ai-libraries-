from enum import Enum

class JobStatus(str, Enum):
    COMPLETED = "completed"
    DOWNLOADING = "downloading"
    FAILED = "failed"
    PENDING = "pending"
    PROCESSING = "processing"

    def __str__(self) -> str:
        return str(self.value)
