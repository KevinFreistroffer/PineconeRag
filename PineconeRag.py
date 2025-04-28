import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeAsyncio, ServerlessSpec
from enum import Enum
from pprint import pprint
from typing import TypedDict, Optional

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")


class PineconeRag:
    def __init__(self, configs):
        self.pinecone_configs = configs
        pprint(self.pinecone_configs)

    # def embedder(self, file, configs):
    #    print(file)
    #    print(configs)
