"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ProductMatch(BaseModel):
    """Product match result"""
    type: str = Field(..., description="Fashion item type (e.g., 'dress', 'top')")
    color: str = Field(..., description="Dominant color name")
    match_type: str = Field(..., description="Match type: 'exact', 'similar', or 'no_match'")
    matched_product_id: str = Field(..., description="Matched product ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Similarity confidence score")
    product_name: Optional[str] = Field(None, description="Product name")


class VideoProcessRequest(BaseModel):
    """Request schema for video processing"""
    video_id: str = Field(..., description="Unique video identifier")
    video_path: str = Field(..., description="Path to video file")
    caption: Optional[str] = Field("", description="Video caption text")
    hashtags: Optional[str] = Field("", description="Hashtags string")


class VideoProcessResponse(BaseModel):
    """Response schema for video processing"""
    video_id: str = Field(..., description="Video identifier")
    vibes: List[str] = Field(..., description="List of detected vibes")
    products: List[ProductMatch] = Field(..., description="List of matched products")


class BuildIndexRequest(BaseModel):
    """Request schema for building FAISS index"""
    catalog_path: Optional[str] = Field(None, description="Path to catalog CSV file")
    force_rebuild: bool = Field(False, description="Force rebuild even if index exists")


class BuildIndexResponse(BaseModel):
    """Response schema for index building"""
    success: bool = Field(..., description="Whether index was built successfully")
    message: str = Field(..., description="Status message")
    num_products: Optional[int] = Field(None, description="Number of products indexed")


