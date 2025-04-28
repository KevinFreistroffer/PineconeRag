import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from enum import Enum
from typing import TypedDict, Optional
from PineconeRag import PineconeRag
import asyncio

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")

async def test():

    file_name = "World-Education-Statistics-2024.pdf"
    root_dir = os.path.dirname(__file__)
    file_path = os.path.join(root_dir, "data_files", file_name)

    configs = {
        "file_configs": {
            # (required)
            "file_name": file_name,
            # (required)
            "file_path": file_path,
            # (required)
            "file_type": "pdf",
            # (optional)
            # Extract data starting on page <int>
            "start_on_page": 0,
            # (optional)
            # Stop extracting data at page <int>
            "end_on_page": None,
        },
        "pinecone_configs": {
            # (required)
            "api_key": PINECONE_API_KEY,

            # (required)
            # use an existing index or create a new index
            "name": "rag-768",

            # (required)
            # The name of the index to create. Must be unique.
            "namespace": "world_education_statistics_2024",

            # (optional)
            # used when creating a new index
            # defaults: 768
            "dimension": 768,
            
            # (optional)
            # Type of similarity metric used in the vector index when querying, one of {"cosine", "dotproduct", "euclidean"}.
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
           
            # (optional)
            # Specify the number of seconds to wait until index gets ready. 
            # defaults: None
            "timeout": None
            
            # (optional)
            # The type of vectors to be stored in the index. One of {"dense", "sparse"}.
        },
    }

    # Test implementation
    rag = PineconeRag(configs=configs)

    # would the user want to get the index details if creating a new index?
    index_details = await rag.ingest()
    print(index_details)

    # answer = rag.prompt("What is the average income in state?")


if __name__ == "__main__":
    asyncio.run(test())
