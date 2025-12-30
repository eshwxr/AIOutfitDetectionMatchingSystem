"""FAISS index builder and searcher for product matching"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
import pandas as pd


class FAISSIndex:
    """FAISS index for fast similarity search"""
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings (512 for ViT-B-32)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []  # Store product metadata (id, name, url, etc.)
        self.is_built = False
    
    def build_index(self, embeddings: np.ndarray, metadata: List[dict]):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Array of normalized embeddings (N x embedding_dim)
            metadata: List of product metadata dicts with keys:
                - product_id: Unique product identifier
                - product_name: Product name
                - image_url: Image URL
                - (optional) other fields
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        # Normalize embeddings (should already be normalized, but ensure)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Create FAISS index (IndexFlatIP for inner product = cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Convert to float32 for FAISS
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Add embeddings to index
        self.index.add(embeddings_f32)
        
        # Store metadata
        self.metadata = metadata
        self.is_built = True
        
        print(f"Built FAISS index with {len(embeddings)} products")
    
    def build_from_catalog(self, catalog_df: pd.DataFrame, encoder, batch_size: int = 32, use_first_image_only: bool = True):
        """
        Build index from catalog DataFrame
        
        Args:
            catalog_df: DataFrame with columns: product_id (or 'id'), product_name (optional), image_url
            encoder: CLIPEncoder instance for encoding images
            batch_size: Batch size for encoding
            use_first_image_only: If True, use only first image per product (default: True)
        """
        # Handle column name variations
        if 'product_id' not in catalog_df.columns and 'id' in catalog_df.columns:
            catalog_df = catalog_df.rename(columns={'id': 'product_id'})
        
        # If multiple images per product, use first image only
        if use_first_image_only:
            catalog_df = catalog_df.groupby('product_id').first().reset_index()
            print(f"Using first image per product. Processing {len(catalog_df)} unique products...")
        
        embeddings_list = []
        metadata_list = []
        
        print(f"Encoding {len(catalog_df)} catalog images...")
        
        # Process in batches
        for i in range(0, len(catalog_df), batch_size):
            batch = catalog_df.iloc[i:i + batch_size]
            
            # Encode batch
            try:
                embeddings_batch = []
                for _, row in batch.iterrows():
                    try:
                        embedding = encoder.encode_image_from_url(row['image_url'])
                        embeddings_batch.append(embedding)
                        metadata_list.append({
                            'product_id': str(row['product_id']),  # Convert to string
                            'product_name': row.get('product_name', f"Product {row['product_id']}"),
                            'image_url': row['image_url']
                        })
                    except Exception as e:
                        print(f"Warning: Failed to encode image for product {row['product_id']}: {e}")
                        continue
                
                if embeddings_batch:
                    embeddings_list.extend(embeddings_batch)
            
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
            
            if (i + batch_size) % 100 == 0:
                print(f"Processed {min(i + batch_size, len(catalog_df))} / {len(catalog_df)} products")
        
        if not embeddings_list:
            raise ValueError("No valid embeddings generated from catalog")
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list)
        
        # Build index
        self.build_index(embeddings, metadata_list)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for similar products
        
        Args:
            query_embedding: Query embedding vector (normalized)
            top_k: Number of top results to return
        
        Returns:
            List of (metadata_dict, similarity_score) tuples, sorted by similarity
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Reshape to (1, embedding_dim)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        similarities, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
        
        # Get results
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(sim)))
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """
        Save index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata pickle
        """
        if not self.is_built:
            raise ValueError("Index not built. Cannot save.")
        
        # Save FAISS index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """
        Load index and metadata from disk
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.embedding_dim = self.index.d
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.is_built = True
        print(f"Loaded index with {len(self.metadata)} products from {index_path}")


