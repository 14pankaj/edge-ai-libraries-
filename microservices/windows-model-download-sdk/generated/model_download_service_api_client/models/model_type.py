from enum import Enum

class ModelType(str, Enum):
    EMBEDDINGS = "embeddings"
    LLM = "llm"
    RERANKER = "reranker"
    VISION = "vision"
    VLM = "vlm"

    def __str__(self) -> str:
        return str(self.value)
