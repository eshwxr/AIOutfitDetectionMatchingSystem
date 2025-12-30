"""Object detection pipeline for fashion items"""

from .frame_extractor import FrameExtractor
from .yolo_detector import YOLODetector
from .bbox_utils import crop_detection, expand_bbox

__all__ = ["FrameExtractor", "YOLODetector", "crop_detection", "expand_bbox"]


