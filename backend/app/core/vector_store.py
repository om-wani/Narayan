import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

# These are module-level variables — created once, reused forever
_embeddings = None
_vector_store = None

client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the local sentence-transformers embedding model.
    First call downloads the model (~90MB) and loads it into memory.
    Every call after that reuses the same loaded model instantly.
    """
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        print("(First run downloads ~90MB — this is a one-time thing)")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embedding model ready.")
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        client = chromadb.PersistentClient(
            path=settings.VECTOR_STORE_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _vector_store = Chroma(
            client=client,
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},  # ← add this
        )
        print(f"Vector store ready at: {settings.VECTOR_STORE_PATH}")
    return _vector_store