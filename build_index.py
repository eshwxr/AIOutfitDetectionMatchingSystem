"""Script to build FAISS index from catalog"""

import sys
import os
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.matching import CLIPEncoder, FAISSIndex
from src.api.main import load_config


def build_index(catalog_path: str = None, force_rebuild: bool = False):
    """Build FAISS index from catalog"""
    config = load_config()
    
    # Determine catalog path
    if catalog_path:
        catalog_path = Path(catalog_path)
    else:
        catalog_path = Path(config['paths']['catalog_dir']) / "catalog.csv"
    
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
    
    # Check if index exists
    index_path = Path(config['matching']['faiss_index_path'])
    metadata_path = Path(config['matching']['catalog_metadata_path'])
    
    if index_path.exists() and not force_rebuild:
        print("Index already exists. Use --force-rebuild to rebuild.")
        return
    
    # Load catalog
    print(f"Loading catalog from {catalog_path}")
    catalog_df = pd.read_csv(catalog_path)
    
    # Handle column name variations (support both 'id' and 'product_id')
    if 'product_id' not in catalog_df.columns and 'id' in catalog_df.columns:
        catalog_df = catalog_df.rename(columns={'id': 'product_id'})
    
    # Validate columns
    required_cols = ['image_url']
    missing_cols = [col for col in required_cols if col not in catalog_df.columns]
    if missing_cols:
        raise ValueError(f"Catalog missing required columns: {missing_cols}")
    
    # Ensure product_id exists (create from index if not present)
    if 'product_id' not in catalog_df.columns:
        catalog_df['product_id'] = catalog_df.index.astype(str)
    
    # Initialize encoder
    print("Initializing CLIP encoder...")
    clip_encoder = CLIPEncoder(
        model_name=config['models']['clip']['model_name'],
        pretrained=config['models']['clip']['pretrained'],
        device=config['models']['clip']['device']
    )
    
    # Build index
    print("Building FAISS index...")
    faiss_index = FAISSIndex(embedding_dim=512)
    faiss_index.build_from_catalog(catalog_df, clip_encoder, batch_size=32)
    
    # Save index
    os.makedirs(index_path.parent, exist_ok=True)
    faiss_index.save(str(index_path), str(metadata_path))
    
    print(f"Index built successfully with {len(faiss_index.metadata)} products")
    print(f"Saved to {index_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from catalog")
    parser.add_argument("--catalog-path", help="Path to catalog CSV file")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if index exists")
    
    args = parser.parse_args()
    
    build_index(catalog_path=args.catalog_path, force_rebuild=args.force_rebuild)

