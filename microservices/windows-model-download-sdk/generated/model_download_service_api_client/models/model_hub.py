from enum import Enum

class ModelHub(str, Enum):
    GETI = "geti"
    HLS = "hls"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENVINO = "openvino"
    PIPELINE_ZOO_MODELS = "pipeline-zoo-models"
    ULTRALYTICS = "ultralytics"

    def __str__(self) -> str:
        return str(self.value)
