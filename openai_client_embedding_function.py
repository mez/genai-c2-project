'''

This file defines a custom embedding function for ChromaDB that uses udacity voc key based client for the OpenAI API to generate embeddings. It is registered with ChromaDB so it can be used in collections.
 
'''

from typing import Dict, Any
from chromadb.api.types import Embeddings, Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import register_embedding_function

@register_embedding_function
class OpenAIClientEmbeddingFunction(EmbeddingFunction):
    def __init__(self, openai_client, model_name: str = "text-embedding-3-small"):
        self.openai_client = openai_client
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        response = self.openai_client.embeddings.create(
            input=input,
            model=self.model_name
        )
        import numpy as np
        return [np.array(data.embedding, dtype=np.float32) for data in response.data]

    @staticmethod
    def name() -> str:
        return "openai-client"

    def get_config(self) -> Dict[str, Any]:
        return dict(model_name=self.model_name)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        raise NotImplementedError("Direct client restoration not supported; pass client at runtime.")
