import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Callable, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not set.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("rag-768")


class Retrieval:
  def __init__(self):
    print('')

  async def query(
      namespace: str,
      text: str,
      top_k: int = 3,
      include_metadata: bool = True,
      include_values: bool = True,
      callback: Optional[Callable[[Dict[str, Any]], None]] = None,
      debug: bool = False,
  ):
    try:
        if not text:
            raise ValueError("Text to query with is required")
        
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        vector = model.encode(text)
        vector_list = vector.tolist()
        response = index.query(
            namespace=namespace,
            vector=vector_list,
            top_k=top_k, 
            include_metadata=include_metadata,
            include_values=include_values
        )
        
        if not response:
            print("No response received from Pinecone")
            return []
            
        if "matches" not in response:
            print("No 'matches' key in response")
            print(f"Response keys: {response.keys() if response else 'None'}")
            return []

        matches = response["matches"]
        
        if not matches:
            print("No matches found in response")
            return []
        
        if callback:
            for match in matches:
                callback(match)
        
        if debug:
            for match in matches:
                id = match["id"]
                score = match["score"]
                values = match["values"]
                metadata = match["metadata"]
                original_text = metadata["original_text"]
                print(f"id: {id}")
                print(f"score: {score}")
                print(f"values: {values}")
                print(f"original_text: {original_text}")

        return matches
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []
