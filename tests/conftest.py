import os

# Set dummy env vars before app import so get_settings() doesn't fail in tests.
os.environ.setdefault("HF_API_TOKEN", "test-token")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("PINECONE_HOST", "https://test.pinecone.io")
