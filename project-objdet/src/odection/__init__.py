"""Object Detection Pytorch Docstring"""

__version__ = "1.0.0"


# from src.model import SSDLite, MobileNetV2
from .dataset_coco import CocoDataset, collate_fn
from .loss import Loss
from .model_evaluate import evaluate
from .ssd import SSD, ResNet
from .transform import SSDTransformer
from .utils import Encoder, coco_classes, generate_dboxes

__all__ = [
    "CocoDataset",
    "collate_fn",
    "Loss",
    "SSD",
    "ResNet",
    "evaluate",
    "SSDTransformer",
    "Encoder",
    "coco_classes",
    "generate_dboxes",
]
