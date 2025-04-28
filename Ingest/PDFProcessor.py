from sentence_transformers import SentenceTransformer
import os
from PyPDF2 import PdfReader
import torch
import asyncio
import tiktoken
from transformers import AutoTokenizer
from pinecone import ServerlessSpec
from enum import Enum
from typing import TypedDict, Optional


class Configs(TypedDict):
    file_name: str
    file_path: str
    file_type: str
    start_on_page: Optional[int]
    end_on_page: Optional[int]


default_configs: Configs = {
    "file_name": None,
    "file_path": None,
    "file_type": None,
    "start_on_page": 0,
    "end_on_page": None,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_tokens(text: str) -> int:
    # encode_plus returns input_ids including special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    encoding = tokenizer.encode_plus(text, add_special_tokens=True)
    return len(encoding["input_ids"])


class PDFProcessor:
    def __init__(self, configs):
        print("Initializing PDFProcessor...")

        # Done in PineconeRag validations?
        if not configs["file_name"]:
            raise ValueError("File name is required")
        try:
            self.configs = configs
            self.raw_text_content = []
            self.embedded_text_content = []
            self.final_records_to_upsert = []  # Final list dict to upsert

            print(
                f"Loading sentence transformer model for file: {self.configs['file_name']}"
            )
            self.model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                device=device,
            )
            print("PDFProcessor initialized successfully")
        except Exception as e:
            print(f"Error initializing PDFProcessor: {e}")
            raise

    def get_reader(self):
        print("Getting PDF reader...")
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pdf_path = os.path.join(root_dir, "data_files", self.configs["file_name"])
        print(f"Attempting to read PDF from: {pdf_path}")
        reader = PdfReader(pdf_path)
        print("PDF reader obtained successfully")
        return reader

    # extracts and stores text_content
    async def extract_text_content(self):
        print("Starting text extraction from PDF...")
        reader = self.get_reader()
        tasks = []
        pages = reader.pages
        start_from = self.configs["start_on_page"] or 0
        end_on = self.configs["end_on_page"] or len(pages)

        if len(pages) == 0:
            raise ValueError("No pages to extract from")

        if start_from > end_on:
            raise ValueError("File config error: start_from cannot be greater than end_on")
        
        for i, page in enumerate(pages[start_from:end_on]):
            if i % 100 == 0:
                print(f"Processing page {i}...")
            task = asyncio.create_task(self.process_page(page))
            tasks.append(task)

        raw_text_content = await asyncio.gather(*tasks)
        self.raw_text_content = self.process_text_across_pages(raw_text_content)
        print(f"Completed text extraction. Total pages processed: {i}")

    async def process_page(self, page):
        text = page.extract_text()
        return text

    def process_text_across_pages(self, raw_text_content):
        if not raw_text_content:
            return []

        processed_content = []
        current_text = raw_text_content[0].strip()
        
        terminal_punctuation = {".", "!", "?", ":", ";"}

        for next_text in raw_text_content[1:]:
            next_text = next_text.strip()

            # If current text ends with terminal punctuation, don't merge
            if current_text and current_text[-1] in terminal_punctuation:
                processed_content.append(current_text)
                current_text = next_text
                continue

            # If next text starts with a capital letter, don't merge
            if next_text and next_text[0].isupper():
                processed_content.append(current_text)
                current_text = next_text
                continue

            # Merge the texts with a space
            current_text = f"{current_text} {next_text}"

        # Add the last piece of text
        if current_text:
            processed_content.append(current_text)
        print(processed_content)
        return processed_content

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
        # self.embedded_text_content = self.model.encode(raw_text_content)
        print("Embedded text content generated successfully")

    def get_text_content(self):
        print("Retrieving text content...")
        return self.raw_text_content

    def get_embeded_text_content(self):
        print("Retrieving embedded text content...")
        return self.embedded_text_content

    def get_pinecone_records(self):
        print("Retrieving Pinecone records...")
        return self.final_records_to_upsert

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
            self.final_records_to_upsert.append(record)

        print(
            f"Prepared {len(self.final_records_to_upsert)} records for Pinecone upsert"
        )

    async def run(self, return_records=False):
        print("Starting PDF processing pipeline...")
        await self.extract_text_content()
        await self.embeded_text_content()
        await self.prepare_records_for_upsert()

        print("PDF processing completed")

        if return_records:
            return self.final_records_to_upsert
