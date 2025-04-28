from .PDFProcessor import PDFProcessor
from .CSVProcessor import CSVProcessor
import asyncio
from enum import Enum
from typing import TypedDict, Optional
from pinecone import PineconeAsyncio, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")

class DeletionProtection(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"

class PineconeConfig(TypedDict):
    name: str
    namespace: str
    host: Optional[str] # 
    dimension: int
    metric: str
    spec: ServerlessSpec
    deletion_protection: Optional[DeletionProtection]
    tags: Optional[dict]

class Embedder():
  def __init__(
      self, 
      # file_configs={
      #   "file_name": "",
      #   "file_type": "pdf",
      #   "start_on_page": 0,
      #   "end_on_page": None,
      # },
      # pinecone_configs={}
       
      configs={
         # File configurations
        "file_name": "",
        "file_type": "pdf",
        "start_on_page": 0,
        "end_on_page": None,

        # Pinecone configurations
        "pinecone": {
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
      }
    ):
    print(self)
    self.configs = configs
  
  async def embed(self):
    file_type = self.configs["file_type"].lower()
    if file_type == "pdf":
      dataset_processor = PDFProcessor()
    elif file_type == "csv":
      dataset_processor = CSVProcessor()

    records = await dataset_processor.run_process()
    print("pinecone_records", len(records))
   
  
  async def upsert_to_pinecone(self, records):
    try:
      async with PineconeAsyncio(api_key=self.configs["pinecone"]["api_key"]) as pc:
        if not await pc.has_index(self.configs["pinecone"]["name"]):
            print("Creating index")
            pc_config = self.configs.copy()
            pc_config.pop("host")
            pc_index = await pc.create_index(**pc_config)
        else:
            if not self.configs["pinecone"]["host"]:
                raise KeyError("PineconeConfig missing 'host' key")
            
            print("Using existing index")
            pc_index = pc.IndexAsyncio(self.configs["host"])

      if records is None or len(records) == 0:
          raise ValueError("Records array is empty.")

      await pc_index.upsert(
          vectors=records,
          namespace=self.configs["pinecone"]["namespace"] if self.configs["pinecone"]["namespace"] else self.configs["file_name"],
          batch_size=100,
      )
      print(f"Successfully upserted all {len(records)} records")
    except Exception as e:
        print(f"Error upserting records: {e}")
        raise
