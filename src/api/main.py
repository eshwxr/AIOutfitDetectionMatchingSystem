"""FastAPI application for FLICKD video processing"""

import os
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    VideoProcessRequest,
    VideoProcessResponse,
    ProductMatch,
    BuildIndexRequest,
    BuildIndexResponse,
)
from src.detection import FrameExtractor, YOLODetector, crop_detection
from src.matching import CLIPEncoder, FAISSIndex, ProductMatcher
from src.classification import VibeClassifier
from src.utils import ColorExtractor, load_video_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FLICKD AI Outfit Detection API",
    description="AI-powered fashion item detection and vibe classification from videos",
    version="1.0.0"
)

# Global variables for models (loaded on startup)
config = None
frame_extractor = None
yolo_detector = None
clip_encoder = None
faiss_index = None
product_matcher = None
vibe_classifier = None
color_extractor = None


def load_config():
    """Load configuration from YAML file"""
    global config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded")


def initialize_models():
    """Initialize all ML models"""
    global frame_extractor, yolo_detector, clip_encoder, faiss_index, product_matcher, vibe_classifier, color_extractor
    
    logger.info("Initializing models...")
    
    # Frame extractor
    frame_extractor = FrameExtractor(
        fps=config['detection']['frame_extraction_fps']
    )
    
    # YOLO detector
    yolo_detector = YOLODetector(
        model_path=config['models']['yolo']['model_path'],
        confidence_threshold=config['models']['yolo']['confidence_threshold'],
        device=config['models']['clip']['device']
    )
    
    # CLIP encoder
    clip_encoder = CLIPEncoder(
        model_name=config['models']['clip']['model_name'],
        pretrained=config['models']['clip']['pretrained'],
        device=config['models']['clip']['device']
    )
    
    # FAISS index
    faiss_index = FAISSIndex(embedding_dim=512)
    
    # Try to load existing index
    index_path = Path(config['matching']['faiss_index_path'])
    metadata_path = Path(config['matching']['catalog_metadata_path'])
    
    if index_path.exists() and metadata_path.exists():
        try:
            faiss_index.load(str(index_path), str(metadata_path))
            logger.info("Loaded existing FAISS index")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    else:
        logger.warning("FAISS index not found. Build index before processing videos.")
    
    # Product matcher
    product_matcher = ProductMatcher(
        faiss_index=faiss_index,
        clip_encoder=clip_encoder,
        exact_threshold=config['matching']['similarity_thresholds']['exact_match'],
        similar_threshold=config['matching']['similarity_thresholds']['similar_match'],
        top_k=config['matching']['top_k']
    )
    
    # Vibe classifier
    vibe_classifier = VibeClassifier(
        clip_encoder=clip_encoder,
        top_k=config['vibes']['top_k_vibes'],
        min_confidence=config['vibes']['min_confidence']
    )
    
    # Color extractor
    color_extractor = ColorExtractor(
        num_colors=config['color']['num_colors']
    )
    
    logger.info("All models initialized")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    load_config()
    initialize_models()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([
            frame_extractor is not None,
            yolo_detector is not None,
            clip_encoder is not None,
            faiss_index is not None,
        ])
    }


@app.post("/build-index", response_model=BuildIndexResponse)
async def build_index(request: BuildIndexRequest):
    """Build or rebuild FAISS index from catalog"""
    global faiss_index, product_matcher
    
    try:
        import pandas as pd
        
        # Determine catalog path
        if request.catalog_path:
            catalog_path = Path(request.catalog_path)
        else:
            catalog_path = Path(config['paths']['catalog_dir']) / "catalog.csv"
        
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
        
        # Check if index exists and force_rebuild is False
        index_path = Path(config['matching']['faiss_index_path'])
        if index_path.exists() and not request.force_rebuild:
            return BuildIndexResponse(
                success=True,
                message="Index already exists. Use force_rebuild=true to rebuild.",
                num_products=None
            )
        
        # Load catalog
        logger.info(f"Loading catalog from {catalog_path}")
        catalog_df = pd.read_csv(catalog_path)
        
        # Handle column name variations (support both 'id' and 'product_id')
        if 'product_id' not in catalog_df.columns and 'id' in catalog_df.columns:
            catalog_df = catalog_df.rename(columns={'id': 'product_id'})
        
        # Validate required columns
        required_cols = ['image_url']
        missing_cols = [col for col in required_cols if col not in catalog_df.columns]
        if missing_cols:
            raise ValueError(f"Catalog missing required columns: {missing_cols}")
        
        # Ensure product_id exists (create from index if not present)
        if 'product_id' not in catalog_df.columns:
            catalog_df['product_id'] = catalog_df.index.astype(str)
        
        # Build index
        logger.info("Building FAISS index...")
        faiss_index = FAISSIndex(embedding_dim=512)
        faiss_index.build_from_catalog(catalog_df, clip_encoder, batch_size=32)
        
        # Save index
        os.makedirs(index_path.parent, exist_ok=True)
        metadata_path = Path(config['matching']['catalog_metadata_path'])
        faiss_index.save(str(index_path), str(metadata_path))
        
        # Reinitialize product matcher
        product_matcher = ProductMatcher(
            faiss_index=faiss_index,
            clip_encoder=clip_encoder,
            exact_threshold=config['matching']['similarity_thresholds']['exact_match'],
            similar_threshold=config['matching']['similarity_thresholds']['similar_match'],
            top_k=config['matching']['top_k']
        )
        
        return BuildIndexResponse(
            success=True,
            message=f"Index built successfully with {len(faiss_index.metadata)} products",
            num_products=len(faiss_index.metadata)
        )
    
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """Process video and return tagged products and vibes"""
    try:
        # Validate video path
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if index is built
        if not faiss_index.is_built:
            raise HTTPException(
                status_code=400,
                detail="FAISS index not built. Please build index first using /build-index endpoint."
            )
        
        # Load caption and hashtags from file if not provided
        caption = request.caption or ""
        hashtags = request.hashtags or ""
        
        if not caption and not hashtags:
            # Try to load from associated .txt file
            loaded_caption, loaded_hashtags = load_video_metadata(str(video_path))
            caption = loaded_caption
            hashtags = loaded_hashtags
        
        logger.info(f"Processing video: {request.video_id}")
        
        # Step 1: Extract frames
        logger.info("Extracting frames...")
        frames = frame_extractor.extract_frames(str(video_path))
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Step 2: Detect fashion items
        logger.info("Detecting fashion items...")
        all_detections = []
        for frame, frame_number in frames:
            detections = yolo_detector.detect(frame, frame_number)
            all_detections.extend(detections)
        
        if not all_detections:
            logger.warning("No fashion items detected in video")
            return VideoProcessResponse(
                video_id=request.video_id,
                vibes=[],
                products=[]
            )
        
        # Step 3: Process each detection
        logger.info(f"Processing {len(all_detections)} detections...")
        products = []
        
        for detection in all_detections:
            # Crop detection
            frame = frames[detection['frame_number']][0]
            cropped = crop_detection(frame, detection['bbox'], expand=True)
            
            # Skip if cropped image is too small
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue
            
            # Extract color
            color_name, _ = color_extractor.extract_dominant_color(cropped)
            
            # Match product
            matches = product_matcher.match_product(cropped)
            
            # Use best match (first one, already sorted by similarity)
            if matches and matches[0]['match_type'] != 'no_match':
                best_match = matches[0]
                product = ProductMatch(
                    type=detection['fashion_type'],
                    color=color_name,
                    match_type=best_match['match_type'],
                    matched_product_id=best_match['matched_product_id'],
                    confidence=best_match['confidence'],
                    product_name=best_match.get('product_name')
                )
                products.append(product)
        
        # Step 4: Classify vibes
        logger.info("Classifying vibes...")
        vibe_results = vibe_classifier.classify(
            caption=request.caption or "",
            hashtags=request.hashtags or ""
        )
        vibes = [vibe for vibe, _ in vibe_results]
        
        logger.info(f"Processing complete: {len(products)} products, {len(vibes)} vibes")
        
        return VideoProcessResponse(
            video_id=request.video_id,
            vibes=vibes,
            products=products
        )
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config['api']['host'] if config else "0.0.0.0",
        port=config['api']['port'] if config else 8000,
        reload=config['api']['reload'] if config else False
    )


