import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeAsyncio, ServerlessSpec
from enum import Enum
from typing import TypedDict, Optional
from PineconeRag import PineconeRag

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")

if __name__ == "__main__":
    rag = PineconeRag(
        configs={
            # (required)
            "api_key": PINECONE_API_KEY,
            # (required)
            # use an existing index or create a new index
            "name": "rag-768",
            # (required)
            # use an existing index or create a new index
            "namespace": "world_education_statistics_2024",
            # (optional)
            # used when creating a new index
            # defaults: 768
            "dimension": 768,
            # (optional)
            # used when creating a new index
            # defaults: cosine
            "metric": "cosine",
            # (optional if using an existing index)
            # verify if an index exists
            "host": "https://rag-768-7c11295.svc.aped-4627-b74a.pinecone.io",
            # (optional)
            # used when creating a new index
            # defaults: ServerlessSpec(cloud="aws", region="us-east-1")
            "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
            # (optional)
            # used when creating a new index
            # defaults: "disabled"
            "deletion_protection": "disabled",
            # (optional)
            # used when creating a new index
            # defaults: {"environment": "development"}
            "tags": {"environment": "development"},
        }
    )
