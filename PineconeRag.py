import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeAsyncio, ServerlessSpec
from enum import Enum
from pprint import pprint
from typing import TypedDict, Optional
from Ingest.CSVProcessor import CSVProcessor
from Ingest.PDFProcessor import PDFProcessor
from Ingest.Ingest import Ingest
from Retrieval.Retrieval import Retrieval

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")


class SupportedFileTypes(Enum):
    PDF = "pdf"
    CSV = "csv"


supported_file_types = [type.value.lower() for type in SupportedFileTypes]


class PineconeRag:
    # add pydantic validation
    def __init__(self, configs):
        self.configs = configs
        self.file_configs = configs["file_configs"]
        self.pinecone_configs = configs["pinecone_configs"]
        self.Embedder = Ingest(configs)
        self.Retrieval = Retrieval(configs)

        if not self.file_configs["file_type"].lower() in supported_file_types:
          raise ValueError(
              f"File type '{self.file_configs['file_type']}' not supported. Supported file types: {', '.join(supported_file_types)}"
          )
        
    async def get_index(self):
      try:
        async with PineconeAsyncio(api_key=self.pinecone_configs["api_key"]) as pc:
          if not await pc.has_index(self.pinecone_configs["name"]):
              print("Creating index")
              pc_config = self.pinecone_configs.copy()
              for k in ( 'api_key', 'host', 'namespace'):
                  pc_config.pop(k, None)
              pc_index = await pc.create_index(**pc_config)
          else:
              if not self.pinecone_configs["host"]:
                  raise KeyError("PineconeConfig missing 'host' key")

              print("Using existing index")
              pc_index = pc.IndexAsyncio(self.pinecone_configs["host"])

              if not pc_index:
                  raise ValueError("Failed to initialize a Pinecone index")
          
          return pc_index
      except Exception as e:
        print(f"Error getting or creating index: {e}")

    async def ingest(self):
        try:
            pc_index = await self.get_index()

            file_type = self.file_configs["file_type"].lower()

            if file_type == SupportedFileTypes.PDF.value:
                print("initializing processor")
                dataset_processor = PDFProcessor(configs=self.file_configs)
            elif file_type == SupportedFileTypes.CSV.value:
                dataset_processor = CSVProcessor(configs=self.file_configs)

            records = await dataset_processor.run(return_records=True)
            # print("pinecone_records", len(records))

            # return records

            # embedder = Embedder(configs=self.configs)
            # records = await self.Embedder.process()

            if not records or len(records) == 0:
                raise ValueError("No records to embed")

            await pc_index.upsert(
                vectors=records,
                namespace=(
                    self.pinecone_configs["namespace"]
                    if self.pinecone_configs["namespace"]
                    else self.file_configs["file_name"]
                ),
                batch_size=100,
            )
            print(f"Successfully upserted all {len(records)} records")

            return pc_index
        except Exception as e:
            print(f"Error upserting records: {e}")
            raise

    async def prompt(self, text: str):
        try:
            print("retrieval " + text)
            #  retrieval = Retrieval()
            return await self.Retrieval.query(text)
        except Exception as e:
            print(f"Error in retrieval: {e}")
            raise
