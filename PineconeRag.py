import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeAsyncio, ServerlessSpec
from enum import Enum
from pprint import pprint
from typing import TypedDict, Optional
from Embedder import Embedder
from Retrieval import Retrieval

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")

class SupportedFileTypes(Enum):
   PDF = "pdf"
   CSV = "csv"

supported_file_types = [type.value.lower() for type in SupportedFileTypes]


class PineconeRag:
  def __init__(self, configs):
    self.configs = configs
    self.file_configs = configs["file_configs"]
    self.pinecone_configs = configs["pinecone_configs"]
    self.Embedder = Embedder(configs)
    self.Retrieval = Retrieval(configs)
    
    if not self.file_configs["file_type"].lower() in supported_file_types:
       raise ValueError(f"File type {self.file_configs["file_type"]} not supported.\n Supported file types are: {', '.join(supported_file_types)}")

  async def embedder(self, file_path):
    print(file_path)    
    try:
      async with PineconeAsyncio(api_key=self.pinecone_configs["api_key"]) as pc:
        if not await pc.has_index(self.pinecone_configs["name"]):
            print("Creating index")
            pc_config = self.pinecone_configs.copy()
            pc_config.pop("host")
            pc_index = await pc.create_index(**pc_config)
        else:
            if not self.pinecone_configs["host"]:
                raise KeyError("PineconeConfig missing 'host' key")
            
            print("Using existing index")
            pc_index = pc.IndexAsyncio(self.pinecone_configs["host"])
      
      # embedder = Embedder(configs=self.configs)
      records = await self.Embedder.embed(file_path=file_path)

      if not records or len(records) == 0:
        raise ValueError("No records to embed")

      if records is None or len(records) == 0:
          raise ValueError("Records array is empty.")

      await pc_index.upsert(
          vectors=records,
          namespace=self.pinecone_configs["namespace"] if self.pinecone_configs["namespace"] else self.file_configs["file_name"],
          batch_size=100,
      )
      print(f"Successfully upserted all {len(records)} records")
    except Exception as e:
        print(f"Error upserting records: {e}")
        raise

  async def retrieval(self, text="What countries have the best education?"):
     try:
         print("retrieval " + text)
        #  retrieval = Retrieval()
         answer = self.Retrieval.query(text)
         return answer
     except Exception as e:
         print(f"Error in retrieval: {e}")
         raise
    


