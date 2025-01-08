import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Qdrant
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", None))
assert QDRANT_GRPC_PORT is not None, "QDRANT_GRPC_PORT environment variable is not set"

QDRANT_HOST = str(os.getenv("QDRANT_HOST", None))
assert QDRANT_HOST is not None, "QDRANT_HOST environment variable is not set"
