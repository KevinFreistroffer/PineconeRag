## Config
```Python
configs = {
  "file_configs": {
      # (required)
      "file_name": file_name,
      # (required)
      "file_path": file_path,
      # (required)
      "file_type": "pdf",
      # (optional)
      "start_on_page": 0,
      # (optional)
      "end_on_page": None,
  },
  "pinecone_configs": {
      # (required)
      "api_key": PINECONE_API_KEY,

      # (required)
      # The name of the index to create. Must be unique within your project and cannot be changed once created. Allowed characters are lowercase letters, numbers, and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
      "name": "rag-768",

      # (required)
      # The name of the index to create. Must be unique.
      "namespace": "world_education_statistics_2024",

      #  If you are creating an index with vector_type="dense" (which is the default), you need to specify dimension to indicate the size of your vectors. This should match the dimension of the embeddings you will be inserting. For example, if you are using OpenAI's CLIP model, you should use dimension=1536. Dimension is a required field when creating an index with vector_type="dense" and should not be passed when vector_type="sparse".
      # defaults: 768
      "dimension": 768,
      
      # Type of similarity metric used in the vector index when querying, one of {"cosine", "dotproduct", "euclidean"}.
      # defaults: cosine
      "metric": "cosine",

      # The type of vectors to be stored in the index. One of {"dense", "sparse"}.
      # defaults: "dense"
      "vector_type": "dense"
      
      # (optional if getting an existing index)
      # verify if an index exists
      "host": "https://rag-768-7c11295.svc.aped-4627-b74a.pinecone.io",
      
      # (optional)
      # A dictionary containing configurations describing how the index should be deployed. For serverless indexes, specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection. Alternatively, use the ServerlessSpec or PodSpec objects to specify these configurations.
      "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
      
      # (optional)
      # If enabled, the index cannot be deleted. If disabled, the index can be deleted.
      # defaults: "disabled"
      "deletion_protection": "disabled",
      
      # (optional)
      # Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
      # defaults: {"environment": "development"}
      "tags": {"environment": "development"},
      
      # (optional)
      # Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds; if -1, return immediately and do not wait.
      # defaults: None
      "timeout": None
  },
}
```