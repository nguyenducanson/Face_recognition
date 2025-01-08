import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from config import QDRANT_GRPC_PORT, QDRANT_HOST
from constance import EMBEDDING_SIZE, QDRANT_COLLECTION_NAME


class Database:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)

        if not self.client.collection_exists(QDRANT_COLLECTION_NAME):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
            )

    def insert_vector(self, vectors: Any):
        if isinstance(vectors, tuple):
            vectors = [vectors]

        self.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid1()),
                    vector=vector,
                    payload={"user_name": name}
                ) for name, vector in vectors
            ]
        )

    def delete_vector(self, ids: Any):
        if isinstance(ids, str):
            ids = [ids]

        self.client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=PointIdsList(points=ids)  # ID of the point to delete
        )

    # def update_vector(self, vectors: Any):
    #     if isinstance(vectors, np.ndarray):
    #         vectors = [vectors]
    #
    #     self.client.upsert(
    #         collection_name=QDRANT_COLLECTION_NAME,
    #         points=[
    #             PointStruct(
    #                 id=idx,  # ID of the point to update
    #                 vector=vector,  # New embedding
    #                 payload={"user_id": idx}
    #             ) for idx, vector in enumerate(vectors)
    #         ]
    #     )

    def search(self, query_vector: np.ndarray):
        search_result = self.client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=1  # Number of closest matches to return
        )
        return search_result


if __name__ == "__main__":
    qdrant_client = Database()

