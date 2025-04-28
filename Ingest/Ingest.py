from .PDFProcessor import PDFProcessor
from .CSVProcessor import CSVProcessor
import asyncio
from enum import Enum
from typing import TypedDict, Optional
from pinecone import PineconeAsyncio, ServerlessSpec

class DeletionProtection(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class PineconeConfig(TypedDict):
    name: str
    namespace: str
    host: Optional[str]  #
    dimension: int
    metric: str
    spec: ServerlessSpec
    deletion_protection: Optional[DeletionProtection]
    tags: Optional[dict]


class Ingest:
    def __init__(self,configs,):
        self.configs = configs
        self.file_configs = configs["file_configs"]
        self.pinecone_configs = configs["pinecone_configs"]

    async def process(self):
        file_type = self.configs["file_configs"]["file_type"].lower()
        if file_type == "pdf":
            dataset_processor = PDFProcessor(configs=self.file_configs)
        elif file_type == "csv":
            dataset_processor = CSVProcessor(configs=self.file_configs)

        records = await dataset_processor.run_process(return_records=True)
        print("pinecone_records", len(records))

        return records

    async def upsert_to_pinecone(self, records):
        try:
            async with PineconeAsyncio(
                api_key=self.pinecone_configs["api_key"]
            ) as pc:
                if not await pc.has_index(self.pinecone_configs["name"]):
                    print("Creating index")
                    pc_config = self.pinecone_configs.copy()
                    pc_config.pop("host")
                    pc_index = await pc.create_index(**pc_config)

                    print("Created pc_index", pc_index)

                    # SHOULD RETURN PC_INDEX to user
                else:
                    if not self.pinecone_configs["host"]:
                        raise KeyError("PineconeConfig missing 'host' key")

                    print("Using existing index")
                    pc_index = pc.IndexAsyncio(self.pinecone_configs["host"])

            if records is None or len(records) == 0:
                raise ValueError("Records array is empty.")

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
        except Exception as e:
            print(f"Error upserting records: {e}")
            raise
