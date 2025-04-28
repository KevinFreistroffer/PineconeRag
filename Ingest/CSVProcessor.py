from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import torch
import asyncio
from transformers import AutoTokenizer
from typing import TypedDict, Optional


class Configs(TypedDict):
    file_name: str
    file_type: str
    text_column: str
    start_row: Optional[int]
    end_row: Optional[int]


default_configs: Configs = {
    "file_name": "",
    "file_type": "csv",
    "text_column": "text",
    "start_row": 0,
    "end_row": None,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_tokens(text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    encoding = tokenizer.encode_plus(text, add_special_tokens=True)
    return len(encoding["input_ids"])


class CSVProcessor:
    def __init__(self, configs: Configs = default_configs):
        print("Initializing CSVProcessor...")
        if not configs["file_name"]:
            raise ValueError("File name is required")
        try:
            self.configs = configs
            self.raw_text_content = []
            self.embedded_text_content = []
            self.final_records_to_upsert = []

            print(
                f"Loading sentence transformer model for file: {self.configs['file_name']}"
            )
            self.model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                device=device,
            )
            print("CSVProcessor initialized successfully")
        except Exception as e:
            print(f"Error initializing CSVProcessor: {e}")
            raise

    def get_reader(self):
        print("Getting CSV reader...")
        root_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(root_dir, "data_files", self.configs["file_name"])
        print(f"Attempting to read CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print("CSV reader obtained successfully")
        return df

    async def extract_text_content(self):
        print("Starting text extraction from CSV...")
        df = self.get_reader()

        if self.configs["text_column"] not in df.columns:
            raise ValueError(f"Column '{self.configs['text_column']}' not found in CSV")

        start_from = self.configs["start_row"] or 0
        end_on = self.configs["end_row"] or len(df)

        print(f"Processing rows from {start_from} to {end_on}...")
        self.raw_text_content = (
            df[self.configs["text_column"]].iloc[start_from:end_on].tolist()
        )
        print(
            f"Completed text extraction. Total rows processed: {len(self.raw_text_content)}"
        )

    async def embeded_text_content(self):
        print("Starting text embedding process...")
        raw_text_content = self.raw_text_content

        if len(raw_text_content) == 0:
            print("No text content to process")
            return

        print("Encoding text content...")
        batch_size = 32
        self.embedded_text_content = []
        for i in range(0, len(raw_text_content), batch_size):
            batch = raw_text_content[i : i + batch_size]
            embeddings = await asyncio.to_thread(self.model.encode, batch)
            self.embedded_text_content.extend(embeddings)
        print("Embedded text content generated successfully")

    async def prepare_records_for_upsert(self):
        print("Preparing records for Pinecone upsert...")

        if len(self.embedded_text_content) == 0:
            print("No embeddings generated")
            raise ValueError("No embeddings to upsert")

        print("Structuring embeddings for upsert...")
        self.final_records_to_upsert = []

        for index, (original_text, embedding) in enumerate(
            zip(self.raw_text_content, self.embedded_text_content)
        ):
            record = {
                "id": str(index),
                "values": embedding,
                "metadata": {"original_text": original_text},
            }
            if index < 5:
                print(record)
            self.final_records_to_upsert.append(record)

        print(
            f"Prepared {len(self.final_records_to_upsert)} records for Pinecone upsert"
        )

    
    def get_text_content(self):
        print("Retrieving text content...")
        return self.raw_text_content

    def get_embeded_text_content(self):
        print("Retrieving embedded text content...")
        return self.embedded_text_content

    def get_pinecone_records(self):
        print("Retrieving Pinecone records...")
        return self.final_records_to_upsert

    async def run(self, return_records=False):
        print("Starting CSV processing pipeline...")
        await self.extract_text_content()
        await self.embeded_text_content()
        await self.prepare_records_for_upsert()

        print("CSV processing completed")

        if return_records:
            return self.get_pinecone_records()
