"""
Vector Database Handler for Qdrant
WITH METADATA SUPPORT FOR PERSISTENT STORAGE
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import uuid


class VectorDB:
    """
    Wrapper for Qdrant Vector Database operations
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
    
    def connect(self):
        """Connect to Qdrant instance"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            print(f"✅ Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int,
        distance: Distance = Distance.COSINE,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a new collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
            metadata: Optional metadata to store with collection (e.g., PDF hash)
        """
        try:
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            # Store metadata if provided
            # Use integer ID 0 for metadata point (Qdrant requires UUID or integer IDs)
            if metadata:
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=0,  # Special ID for metadata
                            vector=[0.0] * vector_size,  # Dummy vector
                            payload={"_metadata": metadata, "_is_metadata": True}
                        )
                    ]
                )
            
            print(f"✅ Collection '{collection_name}' created")
            
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            raise
    
    def get_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata stored with collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Metadata dict or None
        """
        try:
            # Retrieve the special metadata point (ID 0)
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[0]
            )
            
            if result and len(result) > 0:
                return result[0].payload.get("_metadata")
            return None
            
        except:
            return None
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except:
            return False
    
    def insert(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        payloads: List[Dict[str, Any]]
    ):
        """
        Insert vectors with payloads
        
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
        """
        try:
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
                for vector, payload in zip(vectors, payloads)
            ]
            
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            print(f"✅ Inserted {len(points)} vectors into '{collection_name}'")
            
        except Exception as e:
            print(f"❌ Failed to insert vectors: {e}")
            raise
    
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding
            limit: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of results with payload and score
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit + 1,  # Request one extra to account for potential metadata point
                score_threshold=score_threshold
            )
            
            # Format results and filter out metadata point
            formatted_results = []
            for result in results:
                # Skip the metadata point (ID 0 or has _is_metadata flag)
                if result.id == 0 or result.payload.get("_is_metadata"):
                    continue
                    
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    **result.payload
                })
                
                # Stop when we have enough results
                if len(formatted_results) >= limit:
                    break
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✅ Deleted collection '{collection_name}'")
        except Exception as e:
            print(f"⚠️  Could not delete collection: {e}")
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()
            print("✅ Qdrant connection closed")


# Testing
if __name__ == "__main__":
    # Test metadata storage
    vdb = VectorDB()
    vdb.connect()
    
    # Create collection with metadata
    vdb.create_collection(
        "test_collection", 
        vector_size=384,
        metadata={"pdf_hash": "abc123", "created_at": "2024-01-01"}
    )
    
    # Retrieve metadata
    metadata = vdb.get_collection_metadata("test_collection")
    print(f"Stored metadata: {metadata}")
    
    # Cleanup
    vdb.delete_collection("test_collection")
    vdb.close()


# """
# Vector Database (Qdrant)
# Stores document embeddings and retrieves similar chunks
# """
# from typing import List, Dict, Any
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# import uuid

# class VectorDB:
#     """Qdrant Vector Database for semantic search"""
    
#     def __init__(self, host: str, port: int):
#         self.host = host
#         self.port = port
#         self.client = None
    
#     def connect(self):
#         """Connect to Qdrant"""
#         print(f"📡 Connecting to Qdrant at {self.host}:{self.port}...")
#         try:
#             self.client = QdrantClient(host=self.host, port=self.port)
#             print("✅ Connected to Qdrant!")
#         except Exception as e:
#             print(f"❌ Failed to connect to Qdrant: {e}")
#             print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
#             raise
    
#     def create_collection(self, collection_name: str, vector_size: int):
#         """Create collection if it doesn't exist"""
#         try:
#             collections = self.client.get_collections().collections
#             if any(col.name == collection_name for col in collections):
#                 print(f"  Collection '{collection_name}' already exists")
#                 return
            
#             self.client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
#             )
#             print(f"✅ Collection '{collection_name}' created!")
#         except Exception as e:
#             print(f"❌ Error creating collection: {e}")
#             raise
    
#     def insert(self, collection_name: str, vectors: List[List[float]], 
#                payloads: List[Dict[str, Any]]):
#         """Insert vectors with metadata into collection"""
#         ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
#         points = [
#             PointStruct(id=id_, vector=vector, payload=payload)
#             for id_, vector, payload in zip(ids, vectors, payloads)
#         ]
        
#         self.client.upsert(collection_name=collection_name, points=points)
#         print(f"✅ Inserted {len(points)} vectors into '{collection_name}'")
    
#     def search(self, collection_name: str, query_vector: List[float], 
#                limit: int = 3) -> List[Dict[str, Any]]:
#         """Search for similar vectors and return chunks with metadata"""
#         results = self.client.search(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=limit
#         )
        
#         return [
#             {
#                 'score': result.score,
#                 'text': result.payload['text'],
#                 'article': result.payload.get('article_number', 'غير محدد'),
#                 'chapter': result.payload.get('chapter', 'غير محدد'),
#                 'section': result.payload.get('section', 'غير محدد'),
#                 'chunk_id': result.payload.get('chunk_id', 0)
#             }
#             for result in results
#         ]
    
#     def close(self):
#         """Close connection"""
#         if self.client:
#             self.client.close()
#             print("✅ Disconnected from Qdrant")

