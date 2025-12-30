"""Tests for API schemas and validation"""

import pytest
from src.api.schemas import (
    VideoProcessRequest,
    VideoProcessResponse,
    ProductMatch,
    BuildIndexRequest,
    BuildIndexResponse
)


def test_product_match_schema():
    """Test ProductMatch schema"""
    product = ProductMatch(
        type="dress",
        color="black",
        match_type="similar",
        matched_product_id="prod_123",
        confidence=0.85
    )
    
    assert product.type == "dress"
    assert product.color == "black"
    assert product.match_type == "similar"
    assert product.matched_product_id == "prod_123"
    assert product.confidence == 0.85


def test_video_process_request_schema():
    """Test VideoProcessRequest schema"""
    request = VideoProcessRequest(
        video_id="test_123",
        video_path="/path/to/video.mp4",
        caption="Test caption",
        hashtags="#test #fashion"
    )
    
    assert request.video_id == "test_123"
    assert request.video_path == "/path/to/video.mp4"
    assert request.caption == "Test caption"
    assert request.hashtags == "#test #fashion"


def test_video_process_response_schema():
    """Test VideoProcessResponse schema"""
    products = [
        ProductMatch(
            type="dress",
            color="black",
            match_type="similar",
            matched_product_id="prod_123",
            confidence=0.85
        )
    ]
    
    response = VideoProcessResponse(
        video_id="test_123",
        vibes=["Coquette", "Y2K"],
        products=products
    )
    
    assert response.video_id == "test_123"
    assert len(response.vibes) == 2
    assert len(response.products) == 1


def test_build_index_request_schema():
    """Test BuildIndexRequest schema"""
    request = BuildIndexRequest(
        catalog_path="/path/to/catalog.csv",
        force_rebuild=True
    )
    
    assert request.catalog_path == "/path/to/catalog.csv"
    assert request.force_rebuild is True


def test_build_index_response_schema():
    """Test BuildIndexResponse schema"""
    response = BuildIndexResponse(
        success=True,
        message="Index built successfully",
        num_products=100
    )
    
    assert response.success is True
    assert response.message == "Index built successfully"
    assert response.num_products == 100

